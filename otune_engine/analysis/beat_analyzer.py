"""
Beat, tempo, downbeat, time signature, and swing analysis.

Detects tempo (with half/double-time correction), downbeats using
multi-feature importance scoring, time signature, swing ratio,
beat strengths, beat confidence, and musical phrases.
"""

import logging
from typing import List, Tuple

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


class BeatAnalyzer:
    """Comprehensive beat and rhythm analysis."""

    def __init__(self, sample_rate: int = 44100, hop_length: int = 512):
        self.sr = sample_rate
        self.hop = hop_length

    def analyze(self, y_mono: np.ndarray) -> dict:
        """
        Full beat analysis pipeline.

        Returns dict with: tempo, actual_tempo, tempo_multiplier, beats,
        beat_frames, downbeats, time_signature, swing_ratio, groove_type,
        beat_strengths, beat_confidence, phrases, intro_end, outro_start
        """
        # Core beat detection
        tempo, beat_frames = librosa.beat.beat_track(
            y=y_mono, sr=self.sr, hop_length=self.hop, units='frames'
        )
        beats = librosa.frames_to_time(
            beat_frames, sr=self.sr, hop_length=self.hop
        )

        # Onset strength (compute once, reused by multiple methods)
        onset_env = librosa.onset.onset_strength(y=y_mono, sr=self.sr, hop_length=self.hop)

        # Tempo correction (half/double-time)
        actual_tempo, multiplier = self._correct_tempo(
            beats, onset_env, tempo
        )

        # Adjust beat arrays to match corrected tempo so beat density
        # is consistent with actual_tempo throughout downstream processing.
        #   half-time / double-time: keep every other beat
        #   adjusted-up:             interpolate mid-beats
        if multiplier == 'half-time' or multiplier == 'double-time':
            beats = beats[::2]
            beat_frames = beat_frames[::2]
        elif multiplier == 'adjusted-up' and len(beats) >= 2:
            mid_beats = (beats[:-1] + beats[1:]) * 0.5
            beats = np.sort(np.concatenate([beats, mid_beats]))
            mid_frames = ((beat_frames[:-1].astype(np.float64)
                          + beat_frames[1:].astype(np.float64)) * 0.5)
            beat_frames = np.sort(
                np.concatenate([beat_frames.astype(np.float64), mid_frames])
            ).astype(np.int64)

        # Swing / groove detection
        swing_ratio, groove_type = self._detect_swing(y_mono, beats)

        # Time signature
        time_sig = self._detect_time_signature(beats, actual_tempo)

        # Downbeats (measure boundaries)
        downbeats = self._detect_downbeats(
            y_mono, beats, beat_frames, time_sig, actual_tempo
        )

        # Beat strength and confidence
        beat_strengths = self._beat_strengths(beat_frames, onset_env)
        beat_confidence = self._beat_confidence(beats, beat_strengths, actual_tempo)

        # Musical phrases
        phrases = self._detect_phrases(downbeats, actual_tempo, time_sig)

        # Intro/outro boundaries
        duration = len(y_mono) / self.sr
        intro_end, outro_start = self._detect_intro_outro(beats, duration)

        return {
            'tempo': float(tempo.item() if hasattr(tempo, 'item') else tempo),
            'actual_tempo': float(actual_tempo.item() if hasattr(actual_tempo, 'item') else actual_tempo),
            'tempo_multiplier': multiplier,
            'beats': beats,
            'beat_frames': beat_frames,
            'downbeats': downbeats,
            'time_signature': time_sig,
            'swing_ratio': float(swing_ratio),
            'groove_type': groove_type,
            'beat_strengths': beat_strengths,
            'beat_confidence': beat_confidence,
            'phrases': phrases,
            'intro_end': intro_end,
            'outro_start': outro_start,
        }

    # ── Tempo Correction ────────────────────────────────────────────

    def _correct_tempo(self, beats: np.ndarray, onset_env: np.ndarray,
                       detected_tempo: float) -> Tuple[float, str]:
        """Detect and correct half-time / double-time tempo."""
        detected_tempo = float(detected_tempo.item() if hasattr(detected_tempo, 'item') else detected_tempo)
        if len(beats) < 8:
            return detected_tempo, 'normal'

        intervals = np.diff(beats)
        if len(intervals) < 4:
            return detected_tempo, 'normal'

        # Analyze beat strength alternation
        strengths = []
        for bt in beats[:min(30, len(beats))]:
            frame = int(bt * self.sr / self.hop)
            if frame < len(onset_env) - 2:
                strengths.append(np.max(onset_env[max(0, frame - 1):frame + 2]))

        if len(strengths) >= 8:
            arr = np.array(strengths)
            even_avg = np.mean(arr[::2])
            odd_avg = np.mean(arr[1::2])
            ratio = max(even_avg, odd_avg) / (min(even_avg, odd_avg) + 1e-8)

            # Half-time: strong alternating pattern
            if ratio > 1.4:
                actual = detected_tempo / 2
                logger.info(f"  Half-time: {detected_tempo:.0f} → {actual:.0f} BPM")
                return actual, 'half-time'

        # Double-time: fast tempo that should be halved
        if detected_tempo > 140:
            half = detected_tempo / 2
            if 85 <= half <= 110:
                logger.info(f"  Double-time: {detected_tempo:.0f} → {half:.0f} BPM")
                return half, 'double-time'

        # Slow tempo that should be doubled
        if detected_tempo < 70:
            doubled = detected_tempo * 2
            if 90 <= doubled <= 150:
                logger.info(f"  Adjusted up: {detected_tempo:.0f} → {doubled:.0f} BPM")
                return doubled, 'adjusted-up'

        return detected_tempo, 'normal'

    # ── Swing Detection ─────────────────────────────────────────────

    def _detect_swing(self, y: np.ndarray, beats: np.ndarray
                      ) -> Tuple[float, str]:
        """Detect swing ratio and groove type."""
        if len(beats) < 8:
            return 0.5, 'straight'

        hop = self.hop
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=hop)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=self.sr, hop_length=hop
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sr, hop_length=hop)

        ratios = []
        for i in range(min(len(beats) - 1, 40)):
            start, end = beats[i], beats[i + 1]
            dur = end - start
            sub_onsets = onset_times[(onset_times >= start) & (onset_times < end)]
            if len(sub_onsets) >= 2 and dur > 0:
                ratio = (sub_onsets[1] - start) / dur
                if 0.4 < ratio < 0.8:
                    ratios.append(ratio)

        if len(ratios) > 4:
            avg = float(np.median(ratios))
            std = float(np.std(ratios))
            if std < 0.08:
                if 0.58 <= avg <= 0.70:
                    logger.info(f"  Swing groove: ratio {avg:.2f}")
                    return avg, 'swing'
                elif avg > 0.70:
                    logger.info(f"  Shuffle groove: ratio {avg:.2f}")
                    return avg, 'shuffle'
            return avg, 'straight'

        return 0.5, 'straight'

    # ── Time Signature ──────────────────────────────────────────────

    def _detect_time_signature(self, beats: np.ndarray, tempo: float) -> int:
        """Detect beats per measure (2, 3, 4, or 6)."""
        if len(beats) < 8:
            return 4

        intervals = np.diff(beats)
        beat_period = 60.0 / tempo
        best_meter, best_score = 4, 0.0

        for meter in (4, 3, 2, 6):
            n_measures = len(intervals) // meter
            if n_measures < 2:
                continue
            measure_durs = [
                np.sum(intervals[i * meter:(i + 1) * meter])
                for i in range(n_measures)
            ]
            mean_dur = np.mean(measure_durs)
            if mean_dur == 0:
                continue
            cv = np.std(measure_durs) / mean_dur
            consistency = 1.0 if cv < 0.05 else 0.9 if cv < 0.1 else 0.7 if cv < 0.15 else 0.3
            duration_error = abs(mean_dur - beat_period * meter) / (beat_period * meter)
            dur_score = 1.0 if duration_error < 0.05 else 0.8 if duration_error < 0.1 else 0.3
            score = consistency * 0.7 + dur_score * 0.3
            if score > best_score:
                best_score = score
                best_meter = meter

        if best_score > 0.6:
            logger.info(f"  Time signature: {best_meter}/4 (conf: {best_score:.2f})")
        return best_meter

    # ── Downbeats ───────────────────────────────────────────────────

    def _detect_downbeats(self, y: np.ndarray, beats: np.ndarray,
                          beat_frames: np.ndarray, beats_per_measure: int,
                          tempo: float) -> np.ndarray:
        """Detect downbeats using multi-feature importance scoring."""
        if len(beats) < 4:
            return beats

        hop = self.hop
        n_fft = 2048
        S = np.abs(librosa.stft(y, hop_length=hop, n_fft=n_fft))

        # Feature 1: Spectral flux
        flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
        flux = np.pad(flux, (1, 0), mode='edge')

        # Feature 2: Low-frequency energy (bass/kick)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
        low_mask = (freqs >= 20) & (freqs <= 250)
        low_energy = np.sum(S[low_mask, :], axis=0)

        # Feature 3: Onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=hop)

        # Normalize
        def norm(x):
            mx = np.max(x)
            return x / mx if mx > 0 else x

        flux_n = norm(flux)
        low_n = norm(low_energy)
        onset_n = norm(onset_env)

        # Score each beat
        importance = np.zeros(len(beat_frames))
        beat_period = 60.0 / tempo
        expected_db = np.arange(beats[0], beats[-1], beat_period * beats_per_measure)

        for i, bf in enumerate(beat_frames):
            adj = int(bf * self.hop / hop)
            f = flux_n[adj] if adj < len(flux_n) else 0
            lo = low_n[adj] if adj < len(low_n) else 0
            on = onset_n[adj] if adj < len(onset_n) else 0

            # Periodicity score
            min_dist = np.min(np.abs(expected_db - beats[i])) if len(expected_db) > 0 else beat_period
            per = 1.0 - min(min_dist / beat_period, 1.0)

            importance[i] = f * 0.25 + lo * 0.25 + on * 0.30 + per * 0.20

        # Select downbeats
        db_indices = [0]
        for i in range(beats_per_measure, len(beats), beats_per_measure):
            win_start = max(0, i - 1)
            win_end = min(len(importance), i + 2)
            window = importance[win_start:win_end]
            if len(window) > 0:
                db_indices.append(win_start + int(np.argmax(window)))

        # Remove duplicates: overlapping windows with small meter values
        # (e.g. 2/4) can select the same beat for adjacent measures.
        db_indices = list(np.unique(db_indices))
        downbeats = beats[db_indices]
        logger.info(f"  {len(downbeats)} downbeats in {beats_per_measure}/4 time")
        return downbeats

    # ── Beat Strength ───────────────────────────────────────────────

    def _beat_strengths(self, beat_frames: np.ndarray,
                        onset_env: np.ndarray) -> np.ndarray:
        """Multi-feature beat strength scoring (vectorized)."""

        def norm(x):
            mx = np.max(x)
            return x / mx if mx > 0 else x

        onset_n = norm(onset_env)

        strengths = np.zeros(len(beat_frames))
        for i, bf in enumerate(beat_frames):
            if bf < len(onset_n):
                w_start = max(0, bf - 2)
                w_end = min(len(onset_n), bf + 3)
                strengths[i] = np.mean(onset_n[w_start:w_end])

        mx = np.max(strengths)
        if mx > 0:
            strengths /= mx

        # Light smoothing
        if len(strengths) > 5:
            strengths = gaussian_filter1d(strengths, sigma=0.8)
            mx = np.max(strengths)
            if mx > 0:
                strengths /= mx

        return strengths

    # ── Beat Confidence ─────────────────────────────────────────────

    def _beat_confidence(self, beats: np.ndarray,
                         strengths: np.ndarray,
                         tempo: float) -> np.ndarray:
        """Per-beat confidence based on strength, timing, and regularity."""
        if len(beats) < 3 or len(strengths) != len(beats):
            return np.ones(len(beats))

        conf = np.zeros(len(beats))
        period = 60.0 / tempo

        for i in range(len(beats)):
            c = 0.0

            # Relative strength (30%)
            ws = max(0, i - 2)
            we = min(len(strengths), i + 3)
            local = strengths[ws:we]
            mx = np.max(local) if len(local) > 0 else 1.0
            c += (strengths[i] / mx if mx > 0 else 0.5) * 0.30

            # Temporal consistency (40%)
            if i > 0:
                err = abs(beats[i] - beats[i - 1] - period) / period
                c += (1.0 if err < 0.1 else 0.7 if err < 0.2 else 0.3) * 0.40
            else:
                c += 0.30

            # Local regularity (30%)
            if 2 <= i < len(beats) - 2:
                intervals = np.diff(beats[max(0, i - 2):min(len(beats), i + 3)])
                if len(intervals) > 1:
                    cv = np.std(intervals) / (np.mean(intervals) + 1e-8)
                    c += (1.0 if cv < 0.1 else 0.7 if cv < 0.2 else 0.3) * 0.30
                else:
                    c += 0.15
            else:
                c += 0.20

            conf[i] = np.clip(c, 0.0, 1.0)

        return conf

    # ── Phrases ─────────────────────────────────────────────────────

    def _detect_phrases(self, downbeats: np.ndarray, tempo: float,
                        time_sig: int) -> List[Tuple[float, float, int]]:
        """Detect musical phrases (8/16/32 bar sections)."""
        if len(downbeats) < 8:
            return []

        phrases = []
        for phrase_bars in (32, 16, 8):
            if len(downbeats) >= phrase_bars:
                for i in range(0, len(downbeats) - phrase_bars + 1, phrase_bars):
                    end_idx = min(i + phrase_bars, len(downbeats) - 1)
                    phrases.append((
                        float(downbeats[i]),
                        float(downbeats[end_idx]),
                        phrase_bars
                    ))

        phrases.sort(key=lambda x: x[0])
        return phrases

    # ── Intro / Outro ───────────────────────────────────────────────

    def _detect_intro_outro(self, beats: np.ndarray,
                            duration: float) -> Tuple[float, float]:
        """Simple heuristic for intro/outro boundaries."""
        if len(beats) > 32:
            intro_end = float(beats[min(16, len(beats) // 4)])
            outro_start = float(beats[max(-16, -len(beats) // 4)])
        else:
            intro_end = duration * 0.15
            outro_start = duration * 0.85
        return intro_end, outro_start
