"""
Beat alignment and phase correlation for seamless transitions.

Aligns tracks on beat/downbeat/phrase boundaries using
adaptive windowed cross-correlation with genre-specific windows.
"""

import logging
from typing import Tuple, Optional
import numpy as np
from ..core.types import TrackAnalysis

logger = logging.getLogger(__name__)


class BeatAligner:
    """Align beats between two tracks for seamless transitions."""

    def __init__(self, sample_rate: int = 44100, gpu=None,
                 phase_align_window_ms: float = 25.0):
        self.sr = sample_rate
        self.gpu = gpu  # Optional MetalGPU
        self.phase_align_window_ms = float(phase_align_window_ms)

    def align(self, t1: TrackAnalysis, t2: TrackAnalysis,
              audio1: np.ndarray, audio2: np.ndarray,
              crossfade_samples: int,
              intro_skip_samples: int = 0,
              crossfade_start_time: Optional[float] = None,
              t1_time_offset: Optional[float] = None,
              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align audio on beat boundaries for seamless crossfade.

        Tries strategies in priority order:
        1. Phrase boundary alignment (cleanest musical handoff)
        2. Downbeat alignment (measure boundaries)
        3. Strong beat alignment with phase matching (preserve downbeat phase)
        4. Adaptive phase-correlation fine-tuning

        The key improvement over v1: phase matching preserves the
        downbeat/upbeat relationship across the transition, so the
        incoming track's downbeat lands on the outgoing track's
        downbeat, maintaining the groove.
        """
        if crossfade_samples <= 0:
            return audio1, audio2

        beats1_track = t1.beats
        beats2_track = t2.beats
        if len(beats1_track) == 0 or len(beats2_track) == 0:
            return audio1, audio2

        # Compute timing mapping so track1 times line up with the accumulated
        # mix timeline (audio1). We assume audio1 ends at the end of track1.
        duration1 = len(audio1) / self.sr
        xfade_time = crossfade_samples / self.sr
        if duration1 <= 0 or xfade_time <= 0:
            return audio1, audio2

        if t1_time_offset is not None:
            pass  # use the caller-provided offset (accounts for intro_skip)
        else:
            t1_duration = float(getattr(t1, 'duration', 0.0) or 0.0)
            if t1_duration <= 0:
                t1_duration = duration1
            t1_time_offset = duration1 - t1_duration  # mix_time = track_time + offset

        # Adjust track2 timing for intro skip (audio2 is already trimmed,
        # but timing arrays in TrackAnalysis are absolute to the original track)
        skip_time = intro_skip_samples / self.sr

        beats2_shifted_full = beats2_track - skip_time
        beats2_mask = beats2_shifted_full >= 0
        beats2 = beats2_shifted_full[beats2_mask]
        beats2_idx = np.where(beats2_mask)[0]
        if len(beats2) == 0:
            return audio1, audio2

        # Shift track1 timing arrays into mix timeline
        beats1 = beats1_track + t1_time_offset

        downbeats1 = (t1.downbeats + t1_time_offset) if len(t1.downbeats) else np.array([])
        downbeats2_full = (t2.downbeats - skip_time) if len(t2.downbeats) else np.array([])
        downbeats2 = downbeats2_full[downbeats2_full >= 0] if len(downbeats2_full) else np.array([])

        phrases1 = [(float(p[0]) + t1_time_offset, float(p[1]) + t1_time_offset, int(p[2]))
                    for p in (t1.phrases or [])]
        phrases2 = [(float(p[0]) - skip_time, float(p[1]) - skip_time, int(p[2]))
                    for p in (t2.phrases or [])
                    if (float(p[1]) - skip_time) >= 0]

        # Determine desired transition point (start of crossfade) in mix time
        max_start = max(0.0, duration1 - xfade_time)
        if crossfade_start_time is not None and crossfade_start_time > 0:
            tp = float(crossfade_start_time) + t1_time_offset
        else:
            tp = max_start
        tp = float(np.clip(tp, 0.0, max_start))

        # Allow a wider search window for smooth transitions
        tempo = float(getattr(t1, 'actual_tempo', 0.0) or 0.0)
        if tempo > 0.0:
            beat_dur = 60.0 / tempo
            max_early_shift = float(np.clip(beat_dur * 2.0, 0.5, 2.0))
        else:
            max_early_shift = 1.0
        min_time = float(np.clip(tp - max_early_shift, 0.0, max_start))

        # Choose trim points (end_samples in audio1, skip_samples into audio2)
        # Priority: phrase → downbeat → beat-phase
        params = self._phrase_params(audio1, audio2, phrases1, phrases2, tp, xfade_time, max_start, min_time)
        if params is None:
            params = self._downbeat_params(audio1, audio2, downbeats1, downbeats2, tp, xfade_time, max_start, min_time)
        if params is None:
            params = self._beat_params(
                audio1, audio2, beats1, beats2, beats2_idx,
                downbeats1, downbeats2, t1, t2, tp, xfade_time, max_start, min_time
            )

        if params is None:
            return audio1, audio2

        end_samples, skip_samples, reason = params
        end_samples = int(np.clip(end_samples, 1, len(audio1)))
        skip_samples = int(np.clip(skip_samples, 0, max(0, len(audio2) - 1)))

        # Fine phase alignment with wider window for better match
        tuned_skip = self._fine_tune_skip(audio1[:end_samples], audio2, skip_samples, crossfade_samples)
        if tuned_skip != skip_samples:
            offset = int(tuned_skip - skip_samples)
            skip_samples = tuned_skip
            end_samples = int(np.clip(end_samples + offset, int(xfade_time * self.sr), len(audio1)))

        aligned1 = audio1[:end_samples]
        aligned2 = audio2[min(skip_samples, len(audio2) - 1):]
        if len(aligned2) == 0:
            return audio1, audio2
        # Ensure at least 0.5s of post-crossfade audio remains after alignment
        # (accounts for ~5% tempo-ramp compression + crossfade window).
        buffer_samples = max(int(0.5 * self.sr), int(crossfade_samples * 0.1) + 1)
        if len(aligned2) <= crossfade_samples + buffer_samples:
            return audio1, audio2

        # Log in track-local time for readability
        start_mix_t = (len(aligned1) / self.sr) - xfade_time
        start_t1 = start_mix_t - t1_time_offset
        start_t2 = skip_time + (skip_samples / self.sr)
        if reason:
            logger.info(f"  ✓ {reason}-aligned at {start_t1:.1f}s → {start_t2:.1f}s")

        return aligned1, aligned2

    def _phrase_params(self, a1: np.ndarray, a2: np.ndarray,
                       phrases1, phrases2,
                       tp: float, xfade_time: float,
                       max_start: float, min_time: float) -> Optional[Tuple[int, int, str]]:
        """Return (end_samples, skip_samples, reason) for phrase alignment.

        Prefers the LAST phrase end before the transition point for
        the outgoing track, and the FIRST phrase start for the incoming.
        This gives the most natural musical phrasing.
        """
        if not phrases1 or not phrases2:
            return None

        p1_ends = [p[1] for p in phrases1
                   if p[1] <= max_start and p[1] >= min_time]
        if not p1_ends:
            return None
        # Prefer the latest phrase end (most musical handoff)
        best_end = float(max(p1_ends))

        p2_starts = [p[0] for p in phrases2 if 0.0 <= p[0] < 12.0]
        if not p2_starts:
            return None
        # Prefer the earliest phrase start for the incoming track
        best_start = float(min(p2_starts))

        end_samples = int(round((best_end + xfade_time) * self.sr))
        skip_samples = int(round(best_start * self.sr))

        if 0 < end_samples <= len(a1) and 0 <= skip_samples < len(a2):
            return end_samples, skip_samples, 'phrase'
        return None

    def _downbeat_params(self, a1: np.ndarray, a2: np.ndarray,
                         db1: np.ndarray, db2: np.ndarray,
                         tp: float, xfade_time: float,
                         max_start: float, min_time: float) -> Optional[Tuple[int, int, str]]:
        """Return (end_samples, skip_samples, reason) for downbeat alignment.

        Prefers the latest downbeat before tp for outgoing, and the
        earliest downbeat (after intro skip) for incoming — this
        preserves the measure boundary across the transition.
        """
        if len(db1) == 0 or len(db2) == 0:
            return None

        # Latest downbeat before tp for outgoing track
        before = db1[(db1 >= min_time) & (db1 <= tp)]
        if len(before) > 0:
            db1_time = float(before[-1])
        else:
            candidates = db1[(db1 >= min_time) & (db1 <= max_start)]
            if len(candidates) == 0:
                return None
            idx1 = int(np.argmin(np.abs(candidates - tp)))
            db1_time = float(candidates[idx1])
            if abs(db1_time - tp) > 4.0:
                return None

        # Earliest downbeat for incoming track
        after = db2[db2 >= 0]
        db2_time = float(after[0]) if len(after) > 0 else float(db2[0])

        end_samples = int(round((db1_time + xfade_time) * self.sr))
        skip_samples = int(round(db2_time * self.sr))

        if 0 < end_samples <= len(a1) and 0 <= skip_samples < len(a2):
            return end_samples, skip_samples, 'downbeat'
        return None

    def _beat_params(self, a1: np.ndarray, a2: np.ndarray,
                     beats1: np.ndarray, beats2: np.ndarray,
                     beats2_idx: np.ndarray,
                     downbeats1: np.ndarray, downbeats2: np.ndarray,
                     t1: TrackAnalysis, t2: TrackAnalysis,
                     tp: float, xfade_time: float,
                     max_start: float, min_time: float) -> Optional[Tuple[int, int, str]]:
        """Return (end_samples, skip_samples, reason) for strong beat alignment."""
        if len(beats1) == 0 or len(beats2) == 0:
            return None

        # Choose a strong beat close to transition point.
        tempo = float(getattr(t1, 'actual_tempo', 0.0) or 0.0)
        if tempo > 0.0:
            beat_dur = 60.0 / tempo
            window = float(np.clip(beat_dur * 4.0, 2.0, 5.0))
        else:
            window = 3.0

        near = np.where((np.abs(beats1 - tp) < window) & (beats1 <= max_start) & (beats1 >= min_time))[0]
        if len(near) == 0:
            near = np.where((beats1 <= max_start) & (beats1 >= min_time))[0]
        if len(near) == 0:
            return None

        strengths = t1.beat_strengths
        confs = t1.beat_confidence
        valid = near[near < min(len(strengths), len(confs), len(beats1))]
        if len(valid) == 0:
            best_idx = int(near[np.argmin(np.abs(beats1[near] - tp))])
        else:
            distances = np.abs(beats1[valid] - tp)
            dist_w = 1.0 - np.clip(distances / max(window, 1e-6), 0.0, 1.0)
            scores = strengths[valid] * confs[valid] * (0.5 + 0.5 * dist_w)
            best_idx = int(valid[np.argmax(scores)])
        beat1_time = float(beats1[best_idx])

        meter1 = int(getattr(t1, 'time_signature', 4) or 4)
        meter2 = int(getattr(t2, 'time_signature', 4) or 4)

        phase_target = None
        phase2 = None
        # Phase match even across different meters: compute beat phase
        # within a common LCM grid so 4/4 ↔ 3/4 can align.
        if meter1 > 1 and meter2 > 1:
            phase1 = self._beat_phase_indices(beats1, downbeats1, meter1)
            phase2 = self._beat_phase_indices(beats2, downbeats2, meter2)
            if phase1 is None and len(beats1) > 0:
                phase1 = np.arange(len(beats1)) % int(meter1)
            if phase2 is None and len(beats2) > 0:
                phase2 = np.arange(len(beats2)) % int(meter2)
            if phase1 is not None and phase2 is not None and best_idx < len(phase1):
                # Map phase to common grid by finding which beat within the
                # LCM-length super-measure the detected beat corresponds to.
                b1_idx = int(best_idx)
                b1_pos_in_measure = b1_idx % meter1
                # Target phase in common grid = position within the
                # repeated-measure pattern at LCM resolution.
                phase_target = int(b1_pos_in_measure)

        # Choose an early strong beat for track2 (post intro-skip)
        # Extended to 4s max to find better phase-matched entry points
        early_limit = 4.0
        if getattr(t2, 'actual_tempo', 0.0) > 0:
            bar_len = (60.0 / t2.actual_tempo) * max(1, meter2)
            early_limit = float(np.clip(bar_len * 2.0, 1.0, 4.0))

        reason = 'beat'
        early_mask = beats2 < early_limit
        if np.any(early_mask):
            early_local_idx = np.where(early_mask)[0]
            beat2_time = None

            if phase_target is not None and phase2 is not None:
                # Map target phase to track2's meter (modulo)
                phase_target_map = int(phase_target % meter2)
                phase_match = early_local_idx[phase2[early_local_idx] == phase_target_map]
                if len(phase_match) > 0:
                    reason = 'beat-phase'
                    orig_idx = beats2_idx[phase_match]
                    if (len(t2.beat_strengths) == len(t2.beats) and
                            len(t2.beat_confidence) == len(t2.beats)):
                        scores2 = t2.beat_strengths[orig_idx] * t2.beat_confidence[orig_idx]
                        best2_local = int(phase_match[np.argmax(scores2)])
                        beat2_time = float(beats2[best2_local])
                    else:
                        beat2_time = float(beats2[phase_match[0]])

            if beat2_time is None:
                orig_idx = beats2_idx[early_local_idx]
                if (len(t2.beat_strengths) == len(t2.beats) and
                        len(t2.beat_confidence) == len(t2.beats)):
                    scores2 = t2.beat_strengths[orig_idx] * t2.beat_confidence[orig_idx]
                    best2_local = int(early_local_idx[np.argmax(scores2)])
                    beat2_time = float(beats2[best2_local])
                else:
                    beat2_time = float(beats2[early_local_idx[0]])
        else:
            beat2_time = float(beats2[0])

        end_samples = int(round((beat1_time + xfade_time) * self.sr))
        skip_samples = int(round(beat2_time * self.sr))

        if 0 < end_samples <= len(a1) and 0 <= skip_samples < len(a2):
            return end_samples, skip_samples, reason
        return None

    @staticmethod
    def _beat_phase_indices(beats: np.ndarray, downbeats: np.ndarray,
                            meter: int) -> Optional[np.ndarray]:
        """Return per-beat phase (0..meter-1) using downbeat anchors."""
        if meter <= 1 or len(beats) == 0 or len(downbeats) == 0:
            return None

        db_idx = np.array([
            int(np.argmin(np.abs(beats - db))) for db in downbeats
        ], dtype=int)
        if len(db_idx) == 0:
            return None

        db_idx = np.unique(np.clip(db_idx, 0, len(beats) - 1))
        db_idx.sort()

        beat_idx = np.arange(len(beats))
        pos = np.searchsorted(db_idx, beat_idx, side='right') - 1
        pos = np.clip(pos, 0, len(db_idx) - 1)
        phase = (beat_idx - db_idx[pos]) % int(meter)
        return phase

    def _fine_tune_skip(self, a1_trim: np.ndarray, a2_full: np.ndarray,
                        skip_samples: int, crossfade_samples: int) -> int:
        """Micro-align onset phase by adjusting skip_samples within a small window.

        Validates correlation peak against expected lag to avoid locking onto
        a neighboring beat pulse (periodicity-aware peak validation).
        """
        window_samples = int(round(self.phase_align_window_ms * self.sr / 1000.0))
        if window_samples <= 0:
            return skip_samples

        if crossfade_samples <= 0:
            return skip_samples

        start1 = len(a1_trim) - crossfade_samples
        if start1 < 0:
            return skip_samples

        # Use full crossfade region for better correlation
        max_seg = int(self.sr * 2.0)
        seg_len = min(max_seg, crossfade_samples,
                      len(a1_trim) - start1,
                      len(a2_full) - skip_samples)
        if seg_len < 2048:
            return skip_samples

        seg1 = a1_trim[start1:start1 + seg_len]
        seg2 = a2_full[skip_samples:skip_samples + seg_len]

        # Convert to mono for correlation
        if seg1.ndim > 1:
            seg1 = np.mean(seg1, axis=1)
        if seg2.ndim > 1:
            seg2 = np.mean(seg2, axis=1)

        try:
            if self.gpu is not None and hasattr(self.gpu, 'phase_correlation'):
                _, offset = self.gpu.phase_correlation(seg1, seg2, window_samples, self.sr)
            else:
                from scipy import signal as sig
                # Normalize both segments to unit max amplitude (matching GPU behavior)
                # so amplitude differences between tracks don't bias the correlation.
                peak1 = max(np.max(np.abs(seg1)), 1e-8)
                peak2 = max(np.max(np.abs(seg2)), 1e-8)
                corr = sig.correlate(seg1 / peak1, seg2 / peak2, mode='same', method='fft')
                center = len(corr) // 2
                search_start = max(0, center - window_samples)
                search_end = min(len(corr), center + window_samples)
                region = corr[search_start:search_end]
                peak = int(np.argmax(region))
                offset = peak + search_start - center

            # Periodicity-aware peak validation: among peaks within 90% of the
            # max, pick the one closest to center. Prevents jumping to a
            # neighboring pulse (e.g. locking onto a nearby beat).
            peak_val = float(region[peak])
            best_offset = offset
            for candidate_peak in np.argsort(region)[::-1]:
                if region[candidate_peak] < peak_val * 0.9:
                    break
                candidate_offset = candidate_peak + search_start - center
                if abs(candidate_offset) < abs(best_offset):
                    best_offset = candidate_offset
            offset = best_offset

            offset = int(np.clip(offset, -window_samples, window_samples))
            if abs(offset) < 1:
                return skip_samples

            new_skip = int(np.clip(skip_samples + offset, 0, max(0, len(a2_full) - 1)))
            buffer_samples = max(int(0.5 * self.sr), int(crossfade_samples * 0.1) + 1)
            if len(a2_full) - new_skip <= crossfade_samples + buffer_samples:
                return skip_samples

            logger.debug(
                f"  Phase fine-tune: {offset} samples ({offset / self.sr * 1000:.1f}ms)"
            )
            return new_skip

        except Exception:
            return skip_samples


