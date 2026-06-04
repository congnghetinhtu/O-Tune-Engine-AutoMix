"""
Crossfade engine — applies all crossfade styles with vocal awareness.

Supported styles: smooth_blend, energy_punch, harmonic_layer,
build_drop, palate_cleanser, apple_automix.
"""

import logging
import numpy as np
from typing import Optional, Tuple

from ..core.types import TrackAnalysis, TransitionPlan
from .vocal_crossfade import VocalCrossfader
from .transition_ding import TransitionDing

logger = logging.getLogger(__name__)


def _fade_join(a: np.ndarray, b: np.ndarray,
               fade_ms: float = 10.0, sr: int = 44100) -> np.ndarray:
    """Join two arrays with a short cosine crossfade to prevent clicks."""
    n = min(int(fade_ms * sr / 1000.0), len(a), len(b))
    if n < 2:
        return np.concatenate([a, b])
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    fo = np.cos(t * np.pi / 2.0)
    fi = np.sin(t * np.pi / 2.0)
    if a.ndim > 1:
        fo = fo.reshape(-1, 1)
        fi = fi.reshape(-1, 1)
    overlap = a[-n:] * fo + b[:n] * fi
    return np.concatenate([a[:-n], overlap, b[n:]])


class Crossfader:
    """Apply crossfade between aligned audio segments."""

    def __init__(self, sample_rate: int = 44100,
                 vocal_freq_low: float = 300.0,
                 vocal_freq_high: float = 4000.0,
                 duck_db: float = -8.0,
                 ding_enabled: bool = True,
                 ding_level_db: float = -12.0):
        self.sr = sample_rate
        self.vocal_xfade = VocalCrossfader(
            sample_rate, vocal_freq_low, vocal_freq_high, duck_db
        )
        self.ding = TransitionDing(sample_rate, ding_level_db) if ding_enabled else None

    def crossfade(self, audio1: np.ndarray, audio2: np.ndarray,
                  plan: TransitionPlan,
                  t1: Optional[TrackAnalysis] = None,
                  t2: Optional[TrackAnalysis] = None,
                  t1_time_offset: float = 0.0,
                  t2_time_offset: float = 0.0) -> np.ndarray:
        """
        Create crossfaded mix from two aligned audio segments.

        Args:
            audio1: Outgoing track (full length up to transition end).
            audio2: Incoming track (starting from intro skip).
            plan: TransitionPlan with style, duration, etc.
            t1, t2: Track analysis for vocal awareness.
        """
        # Ensure stereo
        audio1 = self._ensure_stereo(audio1)
        audio2 = self._ensure_stereo(audio2)

        xfade_samples = int(plan.crossfade_duration * self.sr)
        style = plan.style

        # Clamp crossfade to actual segment lengths BEFORE vocal detection,
        # so the strategy reflects the crossfade that will actually be used.
        xfade_samples = self._clamp_xfade(xfade_samples, len(audio1), len(audio2))

        # Check for vocal overlap in the transition region.
        # If both tracks have vocals in the critical region, use the
        # vocal-aware frequency-sweep crossfade regardless of style.
        has_v1 = (
            self._has_vocals_in_region(t1, len(audio1) - xfade_samples, len(audio1), t1_time_offset)
            if t1 else False
        )
        has_v2 = (
            self._has_vocals_in_region(t2, 0, xfade_samples, t2_time_offset)
            if t2 else False
        )

        if style in ('energy_punch', 'build_drop') and has_v1 and has_v2:
            logger.info(
                f"  Overriding {style} → frequency_sweep "
                f"(vocals detected in transition region)"
            )
            return self._frequency_sweep_crossfade(
                audio1, audio2, xfade_samples, plan, t1, t2,
                t1_time_offset=t1_time_offset,
                t2_time_offset=t2_time_offset,
                has_v1=has_v1, has_v2=has_v2,
            )

        if style == 'palate_cleanser':
            return self._palate_cleanser(audio1, audio2, xfade_samples, plan)
        elif style == 'energy_punch':
            return self._energy_punch(audio1, audio2, xfade_samples, plan, t1, t2)
        elif style == 'build_drop':
            return self._build_drop(audio1, audio2, xfade_samples, plan)
        else:
            # smooth_blend, harmonic_layer, apple_automix all use the same
            # frequency-sweep crossfade — they differ only via plan.fade_curve_power.
            return self._frequency_sweep_crossfade(
                audio1, audio2, xfade_samples, plan, t1, t2,
                t1_time_offset=t1_time_offset,
                t2_time_offset=t2_time_offset,
                has_v1=has_v1, has_v2=has_v2,
            )

    # ── Frequency Sweep (shared by smooth_blend, harmonic_layer, apple_automix) ─

    def _frequency_sweep_crossfade(self, a1, a2, xfade, plan, t1, t2,
                                    t1_time_offset: float = 0.0,
                                    t2_time_offset: float = 0.0,
                                    has_v1: bool = False,
                                    has_v2: bool = False) -> np.ndarray:
        """Frequency-sweep crossfade with vocal awareness — used by
        smooth_blend, harmonic_layer, and apple_automix styles.
        Differs only by fade_curve_power in the TransitionPlan."""
        xfade = self._clamp_xfade(xfade, len(a1), len(a2))
        if xfade <= 0:
            return np.concatenate([a1, a2])

        end_seg = a1[-xfade:]
        start_seg = a2[:xfade]

        # Gain envelope: ramp track 2 from track 1's level at the start
        # of the overlap down to its natural level by the end.
        # This prevents the post-transition volume jump without applying
        # a constant offset that would mis-match track 2's internal dynamics.
        #
        # Use exponential decay instead of linear so the boost vanishes
        # within the first ~25% of the crossfade. This avoids amplifying
        # phase-vocoder artifacts through most of the overlap, which
        # eliminates the "sudden recover" when the ramp completes.
        rms1 = float(np.sqrt(np.mean(end_seg ** 2)))
        rms2 = float(np.sqrt(np.mean(start_seg ** 2)))
        if rms1 > 1e-8 and rms2 > 1e-8:
            target_gain = float(np.clip(rms1 / rms2, 0.5, 1.5))
            if abs(target_gain - 1.0) > 0.01:
                t = np.linspace(0, 1, xfade)
                ramp = 1.0 + (target_gain - 1.0) * np.exp(-4 * t)
                if start_seg.ndim > 1:
                    ramp = ramp.reshape(-1, 1)
                start_seg = start_seg * ramp

        # For vocal-to-vocal, bump fade curve power to at least 1.0 so
        # the blend is steeper — fat curves (power<1) keep both tracks
        # louder in the middle, doubling vocal overlap.
        fo_power = plan.fade_curve_power
        if has_v1 and has_v2 and fo_power < 1.0:
            fo_power = 1.0
        fo, fi = self._equal_power_curves(xfade, fo_power)
        vocal_strategy = 'bass_swap'

        center_ratio = 0.5
        if has_v1 and has_v2:
            center_ratio = self.vocal_xfade.find_blend_center(end_seg, start_seg)
            if abs(center_ratio - 0.5) > 0.05:
                fo = self.vocal_xfade._warp_fade_center(fo, center_ratio)
                fi = self.vocal_xfade._warp_fade_center(fi, center_ratio)
                logger.info(f"  Phrase-aware blend centered at {center_ratio:.0%}")

        overlap = self.vocal_xfade.crossfade(
            end_seg, start_seg, fo, fi,
            vocal_strategy=vocal_strategy,
            track1_has_vocals=has_v1,
            track2_has_vocals=has_v2,
            fade_curve_power=plan.fade_curve_power,
        )

        result = self._smooth_concat(a1[:-xfade], overlap, a2[xfade:])

        if self.ding is not None and xfade > 0:
            # Compute the crossfade start position in the final output.
            # _smooth_concat blends 30ms (1323 samples) at each join, so
            # the crossfade window starts at len(a1[:-xfade]) - 1323.
            fade_n = int(30.0 * self.sr / 1000.0)
            xfade_start_in_output = max(0, len(a1) - xfade - fade_n)
            lead_samples = min(int(1.0 * self.sr), max(1, xfade_start_in_output))
            ding_pos = xfade_start_in_output - lead_samples
            self.ding.place(result, ding_pos)

        return result

    # ── Energy Punch ────────────────────────────────────────────────

    def _energy_punch(self, a1, a2, xfade, plan, t1, t2) -> np.ndarray:
        """Quick cut with short silence gap for energy jumps."""
        min_post = self._min_post_audio()
        gap_samples = int(plan.gap_duration * self.sr)
        
        # Make fade durations proportional to crossfade duration for natural
        # timing regardless of transition length. Each fade is 15% of the
        # crossfade, clamped to [0.3, 2.0]s for punchy feel.
        fade_out_len = max(int(0.3 * self.sr), min(int(2.0 * self.sr), int(0.15 * xfade), len(a1)))
        fade_in_len = max(int(0.01 * self.sr), min(int(0.15 * xfade), len(a2) - min_post))

        # RMS-based gain: match incoming post-fade level to outgoing pre-fade level
        # Use a short window (3s) just before the fade-out, so the gain reflects
        # what the listener actually hears in the moments before the transition.
        pre_window = max(0, min(int(3.0 * self.sr), len(a1) - fade_out_len))
        if pre_window > 0:
            rms1 = float(np.sqrt(np.mean(a1[-fade_out_len - pre_window:-fade_out_len] ** 2)))
        else:
            rms1 = 0.0
        lookahead = min(max(0, len(a2) - fade_in_len), fade_out_len)
        if lookahead > 0:
            rms2 = float(np.sqrt(np.mean(a2[fade_in_len:fade_in_len + lookahead] ** 2)))
        else:
            rms2 = 0.0
        gain = self._level_gain(rms1, rms2)

        # Quick fade out
        fo = np.power(np.linspace(1, 0, fade_out_len), plan.fade_curve_power)
        fo_2d = fo.reshape(-1, 1)
        faded_end = a1[-fade_out_len:] * fo_2d

        # Short gap
        gap = np.zeros((gap_samples, a1.shape[1]), dtype=a1.dtype) if gap_samples > 0 else np.empty((0, a1.shape[1]))

        # Quick fade in (ends at gain instead of 1.0, so post-fade level matches)
        fi = np.power(np.linspace(0, 1, fade_in_len), 1.0 / plan.fade_curve_power)
        if gain != 1.0:
            fi = fi * gain
        fi_2d = fi.reshape(-1, 1)
        faded_start = a2[:fade_in_len] * fi_2d

        # Apply time-varying gain recovery to remaining incoming track
        remaining = a2[fade_in_len:]
        if gain != 1.0:
            remaining = self._apply_level_recovery(remaining, gain)

        return self._smooth_concat(
            a1[:-fade_out_len], faded_end, gap, faded_start, remaining
        )

    # ── Build Drop ────────────────────────────────────────────────

    def _build_drop(self, a1, a2, xfade, plan) -> np.ndarray:
        """Build-down then quick drop into incoming track."""
        xfade = self._clamp_xfade(xfade, len(a1), len(a2))
        if xfade <= 0:
            return np.concatenate([a1, a2])

        # Allocate a short drop gap; use plan.gap_duration if provided
        if plan.gap_duration > 0:
            drop_len = int(plan.gap_duration * self.sr)
        else:
            drop_len = int(0.1 * xfade)

        drop_len = int(np.clip(drop_len, 0, max(0, xfade - 2)))
        remain = max(2, xfade - drop_len)
        build_len = max(1, int(remain * 0.7))
        rise_len = max(1, remain - build_len)

        build_len = min(build_len, len(a1))
        min_post = self._min_post_audio()
        rise_len = min(rise_len, max(1, len(a2) - min_post))

        # RMS-based gain: match incoming post-fade level to outgoing pre-build level
        # Use a short window (3s) just before the build-out so gain reflects
        # what the listener actually hears before the transition.
        pre_window = max(0, min(int(3.0 * self.sr), len(a1) - build_len))
        if pre_window > 0:
            rms1 = float(np.sqrt(np.mean(a1[-build_len - pre_window:-build_len] ** 2)))
        else:
            rms1 = 0.0
        lookahead = min(max(0, len(a2) - rise_len), int(0.8 * self.sr))
        if lookahead > 0:
            rms2 = float(np.sqrt(np.mean(a2[rise_len:rise_len + lookahead] ** 2)))
        else:
            rms2 = 0.0
        gain = self._level_gain(rms1, rms2)

        fo = np.power(np.linspace(1, 0, build_len), plan.fade_curve_power)
        fi = np.power(np.linspace(0, 1, rise_len), max(0.4, 1.0 / plan.fade_curve_power))
        if gain != 1.0:
            fi = fi * gain

        if a1.ndim > 1:
            fo = fo.reshape(-1, 1)
        if a2.ndim > 1:
            fi = fi.reshape(-1, 1)

        faded_end = a1[-build_len:] * fo
        faded_start = a2[:rise_len] * fi

        gap = np.zeros((drop_len, a1.shape[1]), dtype=a1.dtype) if drop_len > 0 else np.empty((0, a1.shape[1]))

        remaining = a2[rise_len:]
        if gain != 1.0:
            remaining = self._apply_level_recovery(remaining, gain)

        return self._smooth_concat(
            a1[:-build_len], faded_end, gap, faded_start, remaining
        )

    # ── Palate Cleanser ─────────────────────────────────────────────

    def _palate_cleanser(self, a1, a2, xfade, plan) -> np.ndarray:
        """Fade out → silence gap → fade in for clashing keys."""
        min_post = self._min_post_audio()
        gap_samples = int(plan.gap_duration * self.sr)
        fade_len = max(int(0.5 * self.sr), min(int(4.0 * self.sr), int(0.25 * xfade), len(a1), max(min_post, len(a2) - min_post)))

        # RMS-based gain: match incoming post-fade level to outgoing pre-fade level
        # Use a short window (4s) just before the fade-out so gain reflects
        # what the listener actually hears before the transition.
        pre_window = max(0, min(int(4.0 * self.sr), len(a1) - fade_len))
        if pre_window > 0:
            rms1 = float(np.sqrt(np.mean(a1[-fade_len - pre_window:-fade_len] ** 2)))
        else:
            rms1 = 0.0
        lookahead = min(max(0, len(a2) - fade_len), int(2.0 * self.sr))
        if lookahead > 0:
            rms2 = float(np.sqrt(np.mean(a2[fade_len:fade_len + lookahead] ** 2)))
        else:
            rms2 = 0.0
        gain = self._level_gain(rms1, rms2)

        fo = np.power(np.linspace(1, 0, fade_len), plan.fade_curve_power).reshape(-1, 1)
        fi = np.power(np.linspace(0, 1, fade_len), plan.fade_curve_power).reshape(-1, 1)
        if gain != 1.0:
            fi = fi * gain

        gap = np.zeros((gap_samples, a1.shape[1]), dtype=a1.dtype)

        remaining = a2[fade_len:]
        if gain != 1.0:
            remaining = self._apply_level_recovery(remaining, gain)

        return self._smooth_concat(
            a1[:-fade_len], a1[-fade_len:] * fo,
            gap,
            a2[:fade_len] * fi, remaining
        )

    # ── Time-Varying Level Matching ──────────────────────────────────

    @staticmethod
    def _apply_level_recovery(audio: np.ndarray, gain: float,
                               recovery_duration: float = 20.0,
                               sr: float = 44100) -> np.ndarray:
        """Apply a time-varying gain to the incoming track that smoothly
        decays from *gain* back to 1.0 over *recovery_duration* seconds.

        This prevents the permanent level offset that makes the track
        sound artificially boosted or cut after the transition completes.
        """
        if abs(gain - 1.0) < 0.01:
            return audio
        recovery_samples = min(len(audio), int(recovery_duration * sr))
        t_rec = np.linspace(0, 1, recovery_samples, dtype=np.float32)
        recovery = 1.0 + (gain - 1.0) * np.exp(-4 * t_rec)
        if audio.ndim > 1:
            recovery = recovery.reshape(-1, 1)
        tail = audio[:recovery_samples] * recovery
        if recovery_samples < len(audio):
            return np.concatenate([tail, audio[recovery_samples:]])
        return tail

    # ── Utilities ───────────────────────────────────────────────────

    def _min_post_audio(self) -> int:
        """Minimum post-crossfade audio in samples (~0.3s) to prevent abrupt cut."""
        return max(int(0.3 * self.sr), 1)

    def _clamp_xfade(self, xfade: int, len_a1: int, len_a2: int) -> int:
        """Clamp crossfade length, leaving room for post-fade audio."""
        min_post = self._min_post_audio()
        return min(xfade, len_a1, max(1, len_a2 - min_post))

    def _smooth_concat(self, *segments: np.ndarray) -> np.ndarray:
        """Concatenate segments with short cosine fades at each boundary
        to cover the sosfiltfilt startup transient (~20ms) that briefly
        breaks the LR4 reconstruction at the overlap region edges."""
        if len(segments) == 1:
            return segments[0]
        result = segments[0]
        fade_ms = 30.0  # 3× the default — masks band-split filter transients
        for seg in segments[1:]:
            result = _fade_join(result, seg, fade_ms=fade_ms, sr=self.sr)
        return result

    @staticmethod
    def _equal_power_curves(n: int, power: float = 1.0):
        """Generate equal-power fade curves (cos^p, sin^p — power=1 → true equal-power)."""
        t = np.linspace(0, 1, n)
        fo = np.power(np.cos(t * np.pi / 2), power)
        fi = np.power(np.sin(t * np.pi / 2), power)
        return fo, fi

    @staticmethod
    def _level_gain(rms_out: float, rms_in: float) -> float:
        """Compute gain to match incoming track's RMS to outgoing track's pre-fade RMS.
        Clamped to [-6 dB, +3.5 dB] to prevent excessive boosting or cutting.
        Returns 1.0 (no-op) when difference is < 0.1 dB."""
        if rms_out > 1e-8 and rms_in > 1e-8:
            gain = float(np.clip(rms_out / rms_in, 0.5, 1.5))
            if abs(gain - 1.0) > 0.01:
                return gain
        return 1.0

    @staticmethod
    def _ensure_stereo(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            return np.column_stack([audio, audio])
        return audio

    @staticmethod
    def _has_vocals_in_region(track: Optional[TrackAnalysis],
                              start_sample: int, end_sample: int,
                              track_time_offset: float = 0.0) -> bool:
        """Check if track has vocals in sample range.

        track_time_offset maps audio time (seconds) to the original track's
        analysis timeline: track_time = audio_time + track_time_offset.
        """
        if track is None or not track.has_vocals:
            return False
        sr = track.sample_rate
        start_t = (start_sample / sr) + track_time_offset
        end_t = (end_sample / sr) + track_time_offset
        return any(
            v.start < end_t and v.end > start_t
            for v in track.vocal_segments
        )
