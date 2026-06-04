"""
MixPipeline — the main orchestrator for OTune Engine v2.

Coordinates: file discovery → analysis → ordering → transition planning
→ crossfading → output. Each step is a separate module call.
"""

import logging
import platform
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf

from .config import MixConfig
from .types import TrackAnalysis, VocalSegment, StructureSection

from ..acceleration.metal_gpu import MetalGPU
from ..acceleration.parallel import run_parallel_analysis
from ..cache.analysis_cache import AnalysisCache
from ..analysis.audio_loader import (
    load_audio, normalize_loudness, ensure_stereo, to_mono
)
from ..analysis.spectral_analyzer import SpectralAnalyzer
from ..analysis.beat_analyzer import BeatAnalyzer
from ..analysis.key_analyzer import detect_key
from ..analysis.vocal_analyzer import VocalAnalyzer
from ..analysis.structure_analyzer import StructureAnalyzer
from ..analysis.genre_analyzer import detect_genre
from ..analysis.mood_analyzer import estimate_mood
from ..mixing.transition_planner import TransitionPlanner
from ..mixing.track_ordering import TrackOrderer
from ..mixing.crossfader import Crossfader
from ..mixing.beat_aligner import BeatAligner
from ..mixing.tempo_sync import TempoSync

logger = logging.getLogger(__name__)


class MixPipeline:
    """Top-level pipeline orchestrating the complete mixing process."""

    def __init__(self, config: MixConfig):
        self.config = config

        # Initialize GPU
        self.gpu: Optional[MetalGPU] = None
        if config.acceleration.use_gpu:
            self.gpu = MetalGPU()
            if config.acceleration.gpu_batch_size and self.gpu.use_mps:
                self.gpu.batch_size = config.acceleration.gpu_batch_size

        # Initialize cache
        self.cache: Optional[AnalysisCache] = None
        if config.cache.enabled:
            self.cache = AnalysisCache(
                config.cache_path, config.cache.cache_version
            )

        # Initialize analyzers
        sr = config.analysis.sample_rate
        hop = config.analysis.hop_length
        self.spectral = SpectralAnalyzer(sr, config.analysis.n_fft, hop, self.gpu)
        self.beat_analyzer = BeatAnalyzer(sr, hop)
        self.vocal_analyzer = VocalAnalyzer(
            sr, config.analysis.vocal_freq_low,
            config.analysis.vocal_freq_high,
            config.analysis.vocal_min_segment_duration,
            config.analysis.max_analysis_duration,
        )
        self.structure_analyzer = StructureAnalyzer(sr, hop)

        # Initialize mixers
        tc = config.transition
        self.planner = TransitionPlanner(tc)
        self.orderer = TrackOrderer(self.planner)
        self.crossfader = Crossfader(
            sr, config.analysis.vocal_freq_low,
            config.analysis.vocal_freq_high,
            tc.vocal_duck_db,
            ding_enabled=tc.ding_enabled,
            ding_level_db=tc.ding_level_db,
        )
        self.aligner = BeatAligner(sr, self.gpu, tc.beat_align_window_ms)
        self.tempo_sync = TempoSync(sr, tc.tempo_ramp_max_pct)

    # ── Main Entry Point ────────────────────────────────────────────

    def run(self) -> bool:
        """Execute the full mix pipeline. Returns True on success."""
        t0 = time.time()

        # Step 1: Discover audio files
        files = self._discover_files()
        if len(files) < 2:
            logger.error("Need at least 2 audio files to create a mix")
            return False

        logger.info(f"Found {len(files)} audio files")

        # Step 2: Interactive track selection
        custom_order, start_idx = self._interactive_select(files)

        # Step 3: Analyze all tracks
        logger.info("=" * 60)
        logger.info("ANALYZING TRACKS")
        logger.info("=" * 60)
        tracks = self._analyze_all(files)
        if len(tracks) < 2:
            logger.error("Need at least 2 successfully analyzed tracks")
            return False

        # Step 4: Order tracks
        logger.info("=" * 60)
        logger.info("ORDERING TRACKS")
        logger.info("=" * 60)
        if custom_order:
            ordered = self._apply_custom_order(tracks, custom_order, files)
        else:
            ordered = self.orderer.order(tracks, start_idx)

        # Step 5: Mix
        logger.info("=" * 60)
        logger.info("MIXING")
        logger.info("=" * 60)
        result = self._mix_all(ordered)
        if result is None:
            return False

        # Step 6: Final processing and save
        out_dir = self.config.input_folder / "otunedResult"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / self.config.output_file
        self._finalize_and_save(result, output_path)

        elapsed = time.time() - t0
        duration = len(result) / self.config.analysis.sample_rate
        logger.info(f"\n{'=' * 60}")
        logger.info(f"✓ Mix complete!")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Duration: {duration:.1f}s")
        logger.info(f"  Tracks: {len(ordered)}")
        logger.info(f"  Time: {elapsed:.1f}s")
        logger.info(f"{'=' * 60}")
        return True

    # ── File Discovery ──────────────────────────────────────────────

    def _discover_files(self) -> List[Path]:
        """Find all supported audio files in input folder."""
        folder = self.config.input_folder
        files = []
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in self.config.supported_formats and not f.name.startswith('.'):
                files.append(f)
        return files

    # ── Interactive Selection ───────────────────────────────────────

    def _interactive_select(self, files):
        """Let user pick start track or custom order."""
        custom_order = self.config.custom_order
        start_idx = self.config.start_track_index

        if self.config.non_interactive:
            return custom_order, start_idx

        if start_idx is None and custom_order is None and len(files) > 1:
            print(f"\n{'=' * 60}")
            print("Available tracks:")
            print(f"{'=' * 60}")
            for i, f in enumerate(files, 1):
                print(f"  {i}. {f.name}")
            print(f"{'=' * 60}")

            try:
                choice = input(
                    "\nEnter track order (e.g. '3 1 2 4'), "
                    "start track (e.g. '2'), or Enter for auto: "
                ).strip()
                if choice:
                    nums = [int(x) for x in choice.split()]
                    if (len(nums) == len(files) and
                            set(nums) == set(range(1, len(files) + 1))):
                        custom_order = nums
                        print(f"✓ Custom order set\n")
                    elif len(nums) == 1 and 1 <= nums[0] <= len(files):
                        start_idx = nums[0] - 1
                        print(f"✓ Starting with #{nums[0]}\n")
                    else:
                        print("✓ Using auto-selection\n")
                else:
                    print("✓ Using auto-selection\n")
            except (ValueError, KeyboardInterrupt):
                print("\n✓ Using auto-selection\n")

        return custom_order, start_idx

    # ── Analysis ────────────────────────────────────────────────────

    def _analyze_all(self, files: List[Path]) -> List[TrackAnalysis]:
        """Analyze all tracks (parallel or sequential)."""
        workers = self.config.acceleration.max_workers
        use_processes = bool(self.config.acceleration.use_process_pool)
        if self.gpu is not None and getattr(self.gpu, 'use_mps', False):
            use_processes = False

        if platform.system() == 'Darwin' and workers > 1 and use_processes:
            # Process pools can be fragile on macOS (fork + native libs). Prefer threads.
            logger.warning(
                "macOS detected; using threads instead of processes for analysis"
            )
            use_processes = False

        if use_processes and self.cache is not None:
            logger.warning(
                "Disabling cache for process-pool analysis (threading.Lock "
                "is not cross-process safe)"
            )
            self.cache = None

        if workers > 1:
            # ThreadPool by default; ProcessPool only when safe/desired.
            tracks = run_parallel_analysis(
                files, self._analyze_one, workers, use_processes=use_processes
            )
        else:
            tracks = []
            for f in files:
                t = self._analyze_one(f)
                if t is not None:
                    tracks.append(t)

        # Free GPU memory after all analysis completes (not per-track)
        if self.gpu:
            self.gpu.empty_cache()

        return tracks

    def _analyze_one(self, file_path: Path) -> Optional[TrackAnalysis]:
        """Full analysis pipeline for a single track."""
        sr = self.config.analysis.sample_rate

        # Check cache
        if self.cache:
            try:
                cached = self.cache.load(file_path)
                if cached:
                    # Load audio fresh (not cached)
                    audio, _, is_stereo = load_audio(file_path, sr)
                    audio = normalize_loudness(audio, sr, self.config.analysis.target_lufs)
                    t = self._dict_to_track(cached)
                    t.audio_data = audio
                    t.is_stereo = is_stereo
                    return t
            except Exception as e:
                logger.warning(f"  Cache load failed for {file_path.name}: {e}")

        logger.info(f"\nAnalyzing: {file_path.name}")

        try:
            # Load
            audio, _, is_stereo = load_audio(file_path, sr)
            audio = normalize_loudness(audio, sr, self.config.analysis.target_lufs)
            mono = to_mono(audio)
            duration = len(mono) / sr

            # Spectral features (single-pass, GPU-accelerated)
            spec = self.spectral.compute_all(mono)

            # Beat analysis
            beat = self.beat_analyzer.analyze(mono)

            # Key detection
            key_idx, key_name, key_mode, key_conf = detect_key(spec['chroma'])

            # Vocal analysis
            vocal = self.vocal_analyzer.analyze(mono)

            # Structure
            struct = self.structure_analyzer.analyze(
                mono, beat['beats'], spec['chroma'],
                spec['mfccs'], spec['rms'],
                vocal['vocal_segments'],
            )

            # Genre
            genre = detect_genre(
                mono, sr, beat['actual_tempo'],
                spec['energy'], spec['spectral_centroid']
            )

            # Mood (sad → happy)
            mood_score, mood_label = estimate_mood(
                beat['actual_tempo'], spec['energy'], spec['energy_variation'],
                spec['spectral_centroid'], spec['spectral_rolloff'],
                spec['zcr'], key_idx, key_mode, key_conf
            )

            # Build TrackAnalysis
            track = TrackAnalysis(
                file_path=file_path,
                audio_data=audio,
                sample_rate=sr,
                duration=duration,
                is_stereo=is_stereo,
                tempo=beat['tempo'],
                actual_tempo=beat['actual_tempo'],
                tempo_multiplier=beat['tempo_multiplier'],
                beats=beat['beats'],
                beat_frames=beat['beat_frames'],
                downbeats=beat['downbeats'],
                beat_strengths=beat['beat_strengths'],
                beat_confidence=beat['beat_confidence'],
                time_signature=beat['time_signature'],
                swing_ratio=beat['swing_ratio'],
                groove_type=beat['groove_type'],
                key=key_idx,
                key_name=key_name,
                key_mode=key_mode,
                key_confidence=key_conf,
                energy=spec['energy'],
                energy_variation=spec['energy_variation'],
                spectral_centroid=spec['spectral_centroid'],
                spectral_rolloff=spec['spectral_rolloff'],
                spectral_bandwidth=spec['spectral_bandwidth'],
                zcr=spec['zcr'],
                mfcc_mean=spec['mfcc_mean'],
                vocal_segments=vocal['vocal_segments'],
                has_vocals=vocal['has_vocals'],
                structure_sections=struct['structure_sections'],
                main_section=struct['main_section'],
                phrases=beat['phrases'],
                intro_end=beat['intro_end'],
                outro_start=beat['outro_start'],
                genre_hint=genre,
                mood_score=mood_score,
                mood_label=mood_label,
                peak_level=float(np.max(np.abs(mono))),
                rms_level=float(np.sqrt(np.mean(mono ** 2))),
            )

            # Cache (without audio_data)
            if self.cache:
                self.cache.save(file_path, self._track_to_dict(track))

            return track

        except Exception as e:
            logger.error(f"  Analysis failed for {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ── Mixing ──────────────────────────────────────────────────────

    def _mix_all(self, ordered: List[TrackAnalysis]) -> Optional[np.ndarray]:
        """Mix all ordered tracks with transitions and export each pair."""
        mixed = ensure_stereo(ordered[0].audio_data.copy())
        self._log_track_start(ordered[0])

        for i in range(1, len(ordered)):
            t1 = ordered[i - 1]
            t2 = ordered[i]
            self._log_transition_header(i, t1, t2)

            try:
                plan = self.planner.plan(t1, t2)

                audio2, intro_skip_samples = self._apply_intro_skip(t2, plan)

                xfade_samples = int(plan.crossfade_duration * t1.sample_rate)
                t1_mix_offset = self._compute_t1_offset(mixed, t1)
                aligned1, aligned2 = self.aligner.align(
                    t1, t2, mixed, audio2, xfade_samples, intro_skip_samples,
                    crossfade_start_time=plan.crossfade_start,
                )

                t1_time_offset, t2_time_offset = self._compute_time_offsets(
                    t1, t2, audio2, aligned2, plan, intro_skip_samples, t1_mix_offset
                )

                aligned1 = ensure_stereo(aligned1)
                aligned2 = ensure_stereo(aligned2)

                orig_xfade = xfade_samples
                xfade_samples = self._snap_crossfade_to_downbeat(
                    t2, plan, xfade_samples, audio2, aligned1, aligned2, intro_skip_samples
                )

                if xfade_samples != orig_xfade:
                    diff = xfade_samples - orig_xfade
                    new_a1_len = len(aligned1) + diff
                    if 0 < new_a1_len <= len(mixed):
                        aligned1 = ensure_stereo(mixed[:new_a1_len])

                if not (t1.has_vocals and t2.has_vocals):
                    ramp_window = int(xfade_samples * 1.5)
                    aligned1 = self.tempo_sync.apply_ramp(
                        aligned1, t1.actual_tempo, t2.actual_tempo,
                        is_outro=True, ramp_samples=ramp_window,
                    )

                len_a1 = len(aligned1)
                transition_result = self.crossfader.crossfade(
                    aligned1, aligned2, plan, t1, t2,
                    t1_time_offset=t1_time_offset,
                    t2_time_offset=t2_time_offset,
                )

                self._save_pair_transition(
                    transition_result, t1, t2, i, xfade_samples, len_a1
                )

                mixed = transition_result

            except Exception as e:
                logger.error(f"  Transition failed: {e}")
                import traceback
                traceback.print_exc()
                mixed = self._fallback_concat(mixed, t2)

        return mixed

    @staticmethod
    def _log_track_start(track: TrackAnalysis):
        logger.info(f"Starting: {track.file_path.name}")
        logger.info(f"  Tempo: {track.actual_tempo:.1f} BPM | "
                    f"Key: {track.key_name} {track.key_mode} | "
                    f"Genre: {track.genre_hint}")

    @staticmethod
    def _log_transition_header(i: int, t1: TrackAnalysis, t2: TrackAnalysis):
        logger.info(f"\nTransition {i}: {t1.file_path.name} → {t2.file_path.name}")
        logger.info(f"  Tempo: {t1.actual_tempo:.0f} → {t2.actual_tempo:.0f} BPM")
        logger.info(f"  Key: {t1.key_name} {t1.key_mode} → "
                    f"{t2.key_name} {t2.key_mode}")

    @staticmethod
    def _apply_intro_skip(track: TrackAnalysis, plan: TransitionPlan):
        """Apply intro skip to track audio. Returns (trimmed_audio, skip_samples)."""
        audio = track.audio_data.copy()
        skip_samples = int(plan.intro_skip * track.sample_rate)
        if skip_samples > 0:
            if skip_samples >= len(audio):
                skip_samples = max(0, len(audio) - 1)
            if skip_samples > 0:
                audio = audio[skip_samples:]
                logger.info(f"  Skipped {plan.intro_skip:.1f}s intro")
        return audio, skip_samples

    @staticmethod
    def _compute_t1_offset(mixed: np.ndarray, t1: TrackAnalysis) -> float:
        """Compute t1's time offset within the accumulated mix timeline.

        Returns mix_end_t - t1.duration, which maps track_time to mix_time:
            mix_time = track_time + offset
        """
        mix_end_t = len(mixed) / t1.sample_rate
        return mix_end_t - t1.duration

    @staticmethod
    def _compute_time_offsets(t1: TrackAnalysis, t2: TrackAnalysis,
                               audio2: np.ndarray, aligned2: np.ndarray,
                               plan: TransitionPlan, intro_skip_samples: int,
                               t1_mix_offset: float):
        beat_skip_samples = max(0, len(audio2) - len(aligned2))
        t2_time_offset = plan.intro_skip + (beat_skip_samples / t2.sample_rate)
        return -t1_mix_offset, t2_time_offset

    def _snap_crossfade_to_downbeat(self, t2: TrackAnalysis, plan: TransitionPlan,
                                     xfade_samples: int, audio2: np.ndarray,
                                     aligned1: np.ndarray, aligned2: np.ndarray,
                                     intro_skip_samples: int) -> int:
        """Snap crossfade end to track2's nearest downbeat."""
        if len(t2.downbeats) == 0:
            return xfade_samples
        tc = self.config.transition
        beat_skipped = max(0, len(audio2) - len(aligned2))
        xfade_end_t2 = (plan.intro_skip
                        + (beat_skipped + xfade_samples) / t2.sample_rate)
        db_idx = int(np.argmin(np.abs(t2.downbeats - xfade_end_t2)))
        nearest_db = float(t2.downbeats[db_idx])
        new_samples = int(round(
            (nearest_db - plan.intro_skip) * t2.sample_rate - beat_skipped
        ))
        min_xf = int(tc.min_crossfade * t2.sample_rate)
        max_xf = min(len(aligned1), len(aligned2))
        if min_xf <= new_samples <= max_xf:
            old_dur = plan.crossfade_duration
            plan.crossfade_duration = new_samples / t2.sample_rate
            logger.info(
                f"  Crossfade snapped: {old_dur:.1f}s → "
                f"{plan.crossfade_duration:.1f}s "
                f"(lands on track2 downbeat @ {nearest_db:.1f}s)"
            )
            return new_samples
        return xfade_samples

    @staticmethod
    def _limit_audio(audio: np.ndarray, threshold: float = 0.95,
                     lookahead_ms: float = 5.0,
                     attack_ms: float = 1.0,
                     release_ms: float = 200.0,
                     sample_rate: int = 44100) -> np.ndarray:
        """Lookahead limiter with fast attack and slow release.

        Uses a minimum-filter cascade (C-accelerated via scipy.ndimage):
          1. Fast 1ms attack catches transient peaks immediately
          2. Slow release extends gain reduction smoothly

        The lookahead (5ms default) shifts the audio forward so gain
        reduction begins before the peak arrives at the output.
        """
        from scipy.ndimage import minimum_filter1d

        lookahead = int(lookahead_ms * sample_rate / 1000.0)
        attack = max(1, int(attack_ms * sample_rate / 1000.0))
        release = max(1, int(release_ms * sample_rate / 1000.0))
        orig_len = len(audio)
        is_stereo = audio.ndim > 1

        # Pad for lookahead (shift audio forward so gain starts before peak)
        if is_stereo:
            padded = np.pad(audio, ((lookahead, 0), (0, 0)), mode='edge')
        else:
            padded = np.pad(audio, (lookahead, 0), mode='edge')

        env = np.max(np.abs(padded), axis=1) if is_stereo else np.abs(padded)

        required_gain = np.where(
            env > threshold, threshold / np.maximum(env, 1e-8), 1.0
        )

        # Fast attack: minimum filter catches peaks within attack_window (1ms).
        # The min tracks gain drops instantly — as soon as a peak enters the
        # window, the output reflects the lower gain.
        gain_fast = minimum_filter1d(required_gain, size=attack, mode='constant', cval=1.0)

        # Slow release: second min filter extends the reduction over the
        # release window. Even after required_gain returns to 1.0, the
        # smoothed gain stays low until the release window clears.
        smoothed = minimum_filter1d(gain_fast, size=release, mode='constant', cval=1.0)

        if is_stereo:
            for ch in range(audio.shape[1]):
                padded[lookahead:lookahead + orig_len, ch] *= smoothed[:orig_len]
        else:
            padded[lookahead:lookahead + orig_len] *= smoothed[:orig_len]

        return padded[lookahead:lookahead + orig_len]

    @staticmethod
    def _fallback_concat(mixed: np.ndarray, t2: TrackAnalysis) -> np.ndarray:
        """Fallback: simple concatenation with short fade on error."""
        audio2 = ensure_stereo(t2.audio_data.copy())
        fade_len = min(int(1.0 * t2.sample_rate), len(mixed), len(audio2))
        if fade_len > 0:
            t = np.linspace(0, np.pi / 2, fade_len)
            fo = np.cos(t).reshape(-1, 1)
            fi = np.sin(t).reshape(-1, 1)
            overlap = mixed[-fade_len:] * fo + audio2[:fade_len] * fi
            return np.concatenate([
                mixed[:-fade_len], overlap, audio2[fade_len:]
            ])
        return np.concatenate([mixed, audio2])

    # ── Per-Pair Transition Export ──────────────────────────────────

    def _save_pair_transition(self, audio: np.ndarray,
                              t1: TrackAnalysis, t2: TrackAnalysis,
                              idx: int, xfade_samples: int,
                              len_a1: int):
        """Trim the pair transition to a context window and save to file."""
        sr = self.config.analysis.sample_rate
        ctx = max(xfade_samples, int(10 * sr))
        xfade_start = len_a1 - xfade_samples
        trim_start = max(0, xfade_start - ctx)
        trim_end = min(len(audio), xfade_start + xfade_samples + ctx)
        trimmed = audio[trim_start:trim_end]

        out_dir = self.config.input_folder / "otunedResult"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem1 = t1.file_path.stem
        stem2 = t2.file_path.stem
        out_path = out_dir / f"{idx:03d}_{stem1}__{stem2}.wav"
        sf.write(str(out_path), trimmed, sr)
        dur = len(trimmed) / sr
        logger.info(f"  Exported: {out_path.name} ({dur:.1f}s)")

    # ── Finalization ────────────────────────────────────────────────

    def _finalize_and_save(self, audio: np.ndarray, output_path: Path):
        """Normalize, limit, and save final mix."""
        logger.info("Finalizing mix...")

        # Peak limiting with lookahead limiter
        peak = float(np.max(np.abs(audio)))
        if peak > 0.98:
            audio = self._limit_audio(
                audio, threshold=0.98, lookahead_ms=5, release_ms=200,
                sample_rate=self.config.analysis.sample_rate,
            )
            new_peak = float(np.max(np.abs(audio)))
            logger.info(f"  Peak limited: {peak:.3f} → {new_peak:.3f}")

        sf.write(str(output_path), audio, self.config.analysis.sample_rate)

    # ── Custom Order ────────────────────────────────────────────────

    def _apply_custom_order(self, tracks, order, files):
        """Apply user-specified track order."""
        ordered = []
        for idx in order:
            fp = files[idx - 1] if isinstance(idx, int) and 1 <= idx <= len(files) else None
            if fp:
                match = next((t for t in tracks if t.file_path == fp), None)
                if match:
                    ordered.append(match)
        if len(ordered) < 2:
            logger.warning("Custom order failed, using auto")
            return self.orderer.order(tracks)
        return ordered

    # ── Serialization ───────────────────────────────────────────────

    @staticmethod
    def _track_to_dict(t: TrackAnalysis) -> dict:
        """Convert TrackAnalysis to cacheable dict using dataclass fields.
        New fields added to TrackAnalysis are handled automatically."""
        from dataclasses import asdict

        d = asdict(t)

        # Remove audio_data — too large, never cached
        d.pop('audio_data', None)

        # Convert Path to string
        if isinstance(d.get('file_path'), Path):
            d['file_path'] = str(d['file_path'])

        # Convert numpy arrays to lists
        for key, value in list(d.items()):
            if isinstance(value, np.ndarray):
                d[key] = value.tolist()

        # Convert tuples in phrases to lists
        if 'phrases' in d and d['phrases']:
            d['phrases'] = [list(p) for p in d['phrases']]

        return d

    @staticmethod
    def _dict_to_track(d: dict) -> TrackAnalysis:
        """Restore TrackAnalysis from cached dict using dataclass reconstruction.
        New fields added to TrackAnalysis are handled automatically."""
        data = dict(d)

        # Restore Path
        if isinstance(data.get('file_path'), str):
            data['file_path'] = Path(data['file_path'])

        # Restore numpy arrays
        for key in ('beats', 'beat_frames', 'downbeats',
                    'beat_strengths', 'beat_confidence', 'mfcc_mean'):
            if key in data and isinstance(data[key], list):
                data[key] = np.array(data[key])

        # Restore VocalSegment dataclasses
        if 'vocal_segments' in data and isinstance(data['vocal_segments'], list):
            data['vocal_segments'] = [
                VocalSegment(**v) if isinstance(v, dict) else v
                for v in data['vocal_segments']
            ]

        # Restore StructureSection dataclasses
        if 'structure_sections' in data and isinstance(data['structure_sections'], list):
            data['structure_sections'] = [
                StructureSection(**s) if isinstance(s, dict) else s
                for s in data['structure_sections']
            ]

        if 'main_section' in data and isinstance(data['main_section'], dict):
            data['main_section'] = StructureSection(**data['main_section'])

        # Restore phrase tuples
        if 'phrases' in data and isinstance(data['phrases'], list):
            data['phrases'] = [tuple(p) if isinstance(p, list) else p
                               for p in data['phrases']]

        return TrackAnalysis(**data)
