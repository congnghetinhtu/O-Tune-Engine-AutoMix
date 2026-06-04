"""
Vocal-aware crossfading with frequency-sweep blending.

Apple Music-style `bass_swap` / `apple_automix` uses a 3-band frequency
sweep with Linkwitz-Riley 4th-order (LR4) crossover:
  - Low band (LP 300Hz): incoming bass fades in first
  - High band (HP 4000Hz): outgoing highs linger longest
  - Vocal band (300–4000Hz): standard equal-power crossfade

LR4 (24 dB/oct) ensures phase-coherent reconstruction — low + vocal + high
sum to the original with zero phase cancellation, eliminating the "swirly"
artifacts that 1st-order filters produce.
"""

import logging
from typing import Tuple

import numpy as np
from scipy import signal as sig

logger = logging.getLogger(__name__)


class VocalCrossfader:
    """Crossfader with vocal intelligence and clean full-band blending."""

    def __init__(self, sample_rate: int = 44100,
                 vocal_freq_low: float = 300.0,
                 vocal_freq_high: float = 4000.0,
                 duck_db: float = -8.0):
        self.sr = sample_rate
        self.vocal_low = vocal_freq_low
        self.vocal_high = vocal_freq_high
        self.duck_db = duck_db

        # Linkwitz-Riley 4th-order crossover (cascaded 2nd-order Butterworth).
        # Applied twice (cascade) for phase-coherent LR4:
        #   - 24 dB/oct rolloff
        #   - Zero phase difference at crossover → low+vocal+high = original
        #   - No "swirly" artifacts from independent band crossfading
        self._low_sos = sig.butter(
            2, vocal_freq_low, btype='low', output='sos', fs=sample_rate
        )
        self._high_sos = sig.butter(
            2, vocal_freq_high, btype='high', output='sos', fs=sample_rate
        )

    def crossfade(self, audio1: np.ndarray, audio2: np.ndarray,
                  fade_out: np.ndarray, fade_in: np.ndarray,
                  vocal_strategy: str = 'auto',
                  track1_has_vocals: bool = False,
                  track2_has_vocals: bool = False,
                  fade_curve_power: float = 1.0) -> np.ndarray:
        """
        Apply crossfade between two audio segments.

        For 'bass_swap': frequency-sweep crossfade (Apple Music style)
        using LR4 band splitting with unified curves — lows transition
        first, highs linger longest, all bands use the same equal-power
        curve shape shifted in time.

        For 'blend' / 'auto': clean equal-power crossfade on the full
        spectrum — no band splitting, no artifacts.

        For 'duck' / 'gap_align': 3-band vocal-aware crossfade with
        LR4 crossover for phase-coherent vocal isolation.
        """
        n = min(len(audio1), len(audio2), len(fade_out), len(fade_in))
        audio1 = audio1[:n]
        audio2 = audio2[:n]
        fade_out = fade_out[:n]
        fade_in = fade_in[:n]

        if vocal_strategy == 'auto':
            if track1_has_vocals and track2_has_vocals:
                vocal_strategy = 'duck'
            else:
                vocal_strategy = 'blend'

        if vocal_strategy == 'blend':
            return self._standard_crossfade(audio1, audio2, fade_out, fade_in)

        if vocal_strategy == 'bass_swap':
            return self._frequency_sweep_crossfade(
                audio1, audio2, fade_out, fade_in, fade_curve_power,
                track1_has_vocals and track2_has_vocals,
            )

        is_stereo = audio1.ndim > 1
        if is_stereo:
            return self._stereo_vocal_crossfade(
                audio1, audio2, fade_out, fade_in, vocal_strategy
            )
        return self._mono_vocal_crossfade(
            audio1, audio2, fade_out, fade_in, vocal_strategy
        )

    # ── Standard Crossfade (full-band, no splitting) ──────────────

    @staticmethod
    def _standard_crossfade(a1, a2, fo, fi):
        """Clean equal-power crossfade on the full spectrum.
        No band splitting, no limiting — zero artifacts."""
        if a1.ndim > 1:
            fo = fo.reshape(-1, 1)
            fi = fi.reshape(-1, 1)
        return a1 * fo + a2 * fi

    # ── Frequency Sweep Crossfade (bass_swap / apple_automix) ─────

    def _frequency_sweep_crossfade(self, a1, a2, fo, fi, power: float = 1.0,
                                   has_dual_vocals: bool = False):
        """Apple Music-style frequency-sweep crossfade using LR4 band splitting.

        Unified frequency-sweep framework: ALL three bands use the same
        equal-power curve (cos(power) / sin(power)), just shifted in time:
          - Low band: center at t=0.30 (bass transitions first)
          - Vocal band: center at t=0.50 (as specified by caller's fo/fi)
          - High band: center at t=0.70 (highs linger longest)

        This eliminates the spectral imbalance that occurs when each band
        uses a different curve shape — the spectral makeup stays natural
        throughout the entire transition.
        """
        is_stereo = a1.ndim > 1
        if is_stereo:
            n_ch = a1.shape[1]
            result = np.zeros_like(a1)
            for ch in range(n_ch):
                result[:, ch] = self._mono_frequency_sweep(
                    a1[:, ch], a2[:, ch], fo, fi, power, has_dual_vocals
                )
            return result
        return self._mono_frequency_sweep(a1, a2, fo, fi, power, has_dual_vocals)

    def _mono_frequency_sweep(self, a1, a2, fo, fi, power: float = 1.0,
                              has_dual_vocals: bool = False):
        """Mono frequency-sweep with LR4 band splitting and unified curves."""
        low1, voc1, high1 = self._split_bands(a1)
        low2, voc2, high2 = self._split_bands(a2)

        n = len(a1)
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)

        # ── Per-band level matching ──────────────────────────────────
        # Match the incoming low-band RMS to the outgoing band so the
        # bass energy stays consistent through the transition instead of
        # dipping when one track has quieter low end than the other.
        def _match_rms(src, tgt):
            rs = float(np.sqrt(np.mean(src ** 2)))
            rt = float(np.sqrt(np.mean(tgt ** 2)))
            if rs > 1e-8 and rt > 1e-8:
                gain = np.clip(rs / rt, 0.5, 1.5)
                if abs(gain - 1.0) > 0.01:
                    return tgt * gain
            return tgt

        low2 = _match_rms(low1, low2)

        # Unified approach: same curve shape (cos/sin), different center shifts
        # High band shifts left (transitions first: center at ~27.5%)
        # The outgoing highs clear out early so the incoming highs
        # can walk in without clashing with the previous track's top end.
        high_t = np.clip(t / 0.55, 0.0, 1.0)
        high_fo = np.power(np.maximum(np.cos(high_t * np.pi / 2.0), 0.0), power)
        high_fi = np.power(np.maximum(np.sin(high_t * np.pi / 2.0), 0.0), power)
        high_mix = high1 * high_fo + high2 * high_fi

        # Vocal band: original caller-provided curves (center at 50%)
        # When both tracks have vocals, duck the outgoing track so the
        # incoming vocal stays clear — prevents vocal clash during the blend.
        if has_dual_vocals:
            duck_factor = 10 ** (float(self.duck_db) / 20.0)
            duck = 1.0 - (1.0 - duck_factor) * fi
            duck = np.clip(duck, duck_factor, 1.0)
            voc_mix = voc1 * fo * duck + voc2 * fi
        else:
            voc_mix = voc1 * fo + voc2 * fi

        # Low band shifts right (transitions later: center at ~60%)
        # The outgoing bass sustains through the transition so the
        # low-end energy doesn't drop — the listener feels the groove
        # continue while the highs and mids transition above it.
        low_t = np.clip((t - 0.20) / 0.80, 0.0, 1.0)
        low_fo = np.power(np.maximum(np.cos(low_t * np.pi / 2.0), 0.0), power)
        low_fi = np.power(np.maximum(np.sin(low_t * np.pi / 2.0), 0.0), power)
        low_mix = low1 * low_fo + low2 * low_fi

        return low_mix + voc_mix + high_mix

    # ── 3-Band Vocal Crossfade (duck / gap_align) ────────────

    def _mono_vocal_crossfade(self, a1, a2, fo, fi, strategy: str):
        low1, voc1, high1 = self._split_bands(a1)
        low2, voc2, high2 = self._split_bands(a2)

        low_mix = low1 * fo + low2 * fi
        high_mix = high1 * fo + high2 * fi

        if strategy == 'duck':
            voc_mix = self._duck_vocal_band(voc1, voc2, fo, fi)
        elif strategy == 'gap_align':
            voc_mix = self._gap_align_vocal_band(voc1, voc2, fo, fi)
        else:
            voc_mix = voc1 * fo + voc2 * fi

        return low_mix + voc_mix + high_mix

    def _stereo_vocal_crossfade(self, a1, a2, fo, fi, strategy: str):
        n_channels = a1.shape[1]
        result = np.zeros_like(a1)
        for ch in range(n_channels):
            result[:, ch] = self._mono_vocal_crossfade(
                a1[:, ch], a2[:, ch], fo, fi, strategy
            )
        return result

    # ── Band Splitting (LR4, phase-coherent) ──────────────────────

    def _split_bands(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Linkwitz-Riley 4th-order crossover (24 dB/oct).

        Uses zero-phase forward-backward filtering (sosfiltfilt) so the
        LR4 crossover introduces no group delay — transients stay punchy
        and the outgoing track never sounds "laggy" during a transition.
        """
        audio = audio.astype(np.float32)
        low = sig.sosfiltfilt(self._low_sos, audio, axis=0)
        high = sig.sosfiltfilt(self._high_sos, audio, axis=0)
        vocal = audio - low - high
        return low, vocal, high

    # ── Ducking Strategy ────────────────────────────────────────────

    def _duck_vocal_band(self, voc1: np.ndarray, voc2: np.ndarray,
                         fo: np.ndarray, fi: np.ndarray) -> np.ndarray:
        """
        Gentle ducking of the outgoing vocal to make room for the incoming.

        Only the outgoing vocal (voc1) is ducked — the incoming vocal plays
        at its full natural level so it stays present and continuous through
        the transition. The duck follows a slow energy envelope (~300 ms)
        to glide smoothly rather than gate.
        """
        n = len(voc1)

        db_val = float(self.duck_db)
        if db_val >= 0:
            db_val = -8.0
        duck_factor = 10 ** (db_val / 20.0)

        e2 = self._frame_energy(voc2, window=2048)
        if len(e2) < 2:
            return voc1 * fo + voc2 * fi

        t = np.linspace(0, len(e2) - 1, n)
        env = np.interp(t, np.arange(len(e2)), e2.astype(np.float32))
        env = env / max(np.max(env), 1e-8)

        from scipy.ndimage import gaussian_filter1d
        sigma = max(1, int(0.3 * self.sr / 4))
        env = gaussian_filter1d(env, sigma=sigma)

        duck = 1.0 - (1.0 - duck_factor) * env * fo
        duck = np.clip(duck, duck_factor, 1.0)

        voc_mix = voc1 * fo * duck + voc2 * fi
        return voc_mix

    # ── Gap Alignment Strategy ──────────────────────────────────────

    def _gap_align_vocal_band(self, voc1: np.ndarray, voc2: np.ndarray,
                              fo: np.ndarray, fi: np.ndarray) -> np.ndarray:
        """
        Align the crossfade center to a vocal gap.

        Shifts the effective crossfade point to where outgoing vocals
        are silent, creating the cleanest possible transition.
        Falls back to ducking if no good gap is found.

        Uses a wider frame window and softer acceptance threshold so
        gaps are found more reliably, and applies a gentle blend
        around the gap edges to avoid abrupt transitions.
        """
        e1 = self._frame_energy(voc1, window=4096)
        e2 = self._frame_energy(voc2, window=4096)

        n_frames = min(len(e1), len(e2))

        # Find lowest-energy region (potential gap) using combined energy
        if n_frames > 10:
            from scipy.ndimage import gaussian_filter1d
            e1_smooth = gaussian_filter1d(e1[:n_frames], sigma=6)
            e2_smooth = gaussian_filter1d(e2[:n_frames], sigma=6)
            combined = e1_smooth + e2_smooth

            center_start = len(combined) // 5
            center_end = 4 * len(combined) // 5
            center_region = combined[center_start:center_end]

            if len(center_region) > 0:
                gap_idx = int(center_start + np.argmin(center_region))
                denom = max(1, len(combined) - 1)
                gap_ratio = float(gap_idx / denom)

                # Softer threshold to catch gaps more reliably
                threshold = max(np.mean(combined) * 0.45, np.median(combined) * 0.6)
                if combined[gap_idx] < threshold:
                    shifted_fo = self._warp_fade_center(fo, gap_ratio)
                    shifted_fi = self._warp_fade_center(fi, gap_ratio)
                    logger.debug(f"  Gap-aligned at {gap_ratio:.0%}")
                    return voc1 * shifted_fo + voc2 * shifted_fi

        # Fallback to ducking
        return self._duck_vocal_band(voc1, voc2, fo, fi)

    # ── Phrase-Aware Blend Center ────────────────────────────────────

    def find_blend_center(self, audio1: np.ndarray,
                          audio2: np.ndarray) -> float:
        """Find the optimal center point for a blend within a crossfade window.

        Analyzes vocal-band energy of both segments and returns a ratio [0,1]
        where the blend midpoint should land — typically at a vocal gap or
        phrase boundary. Falls back to 0.5 (center) when no clear gap exists.
        """
        n = min(len(audio1), len(audio2))
        if n < self.sr // 2:
            return 0.5

        _, voc1, _ = self._split_bands(audio1[:n])
        _, voc2, _ = self._split_bands(audio2[:n])

        e1 = self._frame_energy(voc1, window=2048)
        e2 = self._frame_energy(voc2, window=2048)

        n_frames = min(len(e1), len(e2))
        if n_frames < 10:
            return 0.5

        from scipy.ndimage import gaussian_filter1d
        sigma_frames = max(1, int(0.15 * self.sr / 2048))
        e1_s = gaussian_filter1d(e1[:n_frames], sigma=sigma_frames)
        e2_s = gaussian_filter1d(e2[:n_frames], sigma=sigma_frames)
        combined = e1_s + e2_s

        lo = max(1, int(0.15 * n_frames))
        hi = min(n_frames - 1, int(0.85 * n_frames))
        region = combined[lo:hi]

        gap_idx = int(np.argmin(region))
        gap_val = region[gap_idx]
        mean_val = float(np.mean(region))

        if gap_val < mean_val * 0.7:
            ratio = (lo + gap_idx) / max(1, n_frames - 1)
            return float(np.clip(ratio, 0.3, 0.7))

        energy_drops = np.diff(combined, prepend=combined[0])
        best_drop = int(np.argmin(energy_drops[lo:hi])) + lo
        drop_ratio = best_drop / max(1, n_frames - 1)
        if 0.2 < drop_ratio < 0.8 and energy_drops[best_drop] < -mean_val * 0.2:
            return float(np.clip(drop_ratio, 0.3, 0.7))

        return 0.5

    # ── Utilities ───────────────────────────────────────────────────

    @staticmethod
    def _frame_energy(audio: np.ndarray, window: int = 2048) -> np.ndarray:
        """Compute per-frame RMS energy (handles mono and stereo)."""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        n_frames = len(audio) // window
        if n_frames == 0:
            return np.array([np.sqrt(np.mean(audio ** 2))])
        frames = audio[:n_frames * window].reshape(n_frames, window)
        return np.sqrt(np.mean(frames ** 2, axis=1))

    @staticmethod
    def _warp_fade_center(fade: np.ndarray, center_ratio: float) -> np.ndarray:
        """Time-warp an existing fade so its midpoint lands at center_ratio."""
        n = len(fade)
        if n == 0:
            return fade

        center = float(np.clip(center_ratio, 0.05, 0.95))
        t = np.linspace(0.0, 1.0, n)

        u = np.empty_like(t)
        left = t <= center
        u[left] = t[left] * (0.5 / center)
        u[~left] = 0.5 + (t[~left] - center) * (0.5 / (1.0 - center))

        return np.interp(u, t, fade)
