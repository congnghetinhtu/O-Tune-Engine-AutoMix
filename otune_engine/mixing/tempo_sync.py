"""
Tempo synchronization using librosa's phase vocoder.

Preserves pitch while adjusting timing — like CDJ Master Tempo / Key Lock.
Applied as a single uniform stretch per track (no segmentation), so the
only artifacts come from the phase vocoder itself, which is negligible
for stretch ratios < 2%.

The stretch is applied to the FULL aligned segment before crossfading,
so any residual artifacts land inside the crossfade region where the
frequency-sweep blend naturally masks them.
"""

import logging
import warnings
from typing import Optional

import numpy as np
import librosa

logger = logging.getLogger(__name__)


class TempoSync:
    """Apply tempo ramping via pitch-preserving phase vocoder."""

    def __init__(self, sample_rate: int = 44100, max_change_pct: float = 2.0):
        self.sr = sample_rate
        self.max_pct = max_change_pct

    def apply_ramp(self, audio: np.ndarray, source_tempo: float,
                   target_tempo: float, is_outro: bool = True,
                   ramp_samples: Optional[int] = None) -> np.ndarray:
        """
        Ramp tempo toward target using segmented phase vocoder.

        Only the crossfade-relevant portion is ramped; the rest of the
        audio plays at its natural tempo. The ramp window is controlled
        by *ramp_samples* (typically ~1.5× the crossfade).

        Uses a raised-cosine rate profile so the transition is smooth.
        Each segment is stretched with librosa's pitch-preserving phase
        vocoder, and adjacent segments are crossfaded to mask boundaries.

        For outgoing track: starts at rate=1.0, peaks at end of window.
        For incoming track: peaks at start of window, returns to rate=1.0.
        """
        safe_src = max(float(abs(source_tempo)), 1.0)
        safe_tgt = max(float(abs(target_tempo)), 1.0)
        ratio = max(safe_src, safe_tgt) / min(safe_src, safe_tgt)

        # Skip the ramp when tracks are within 1% tempo — the internal
        # time-compression from the phase vocoder's variable-rate stretch
        # causes more beat drift (~60ms over 8s) than the natural tempo
        # difference it eliminates, making the middle of the crossfade
        # feel rhythmically disconnected despite correct start/end alignment.
        if ratio < 1.01:
            return audio

        # Allow more aggressive ramping for large tempo gaps.
        # Phase-vocoder artifacts from the stretch land inside the
        # crossfade where frequency-sweep blend naturally masks them.
        # At 6% stretch the artifacts are audible on isolated sustained
        # tones but imperceptible in a mixed transition.
        if ratio >= 1.15:
            max_pct = min(self.max_pct * 3.0, 6.0)
        else:
            max_pct = self.max_pct

        diff = abs(source_tempo - target_tempo)
        direction = 1.0 if target_tempo > source_tempo else -1.0
        peak_stretch = direction * min(diff / safe_src, max_pct / 100.0)

        if abs(peak_stretch) < 0.002:
            return audio

        # ── Limit ramp to a window around the transition ─────────────
        # Only the crossfade region needs tempo adjustment; the rest
        # stays at natural tempo to avoid audible drift mid-song.
        if ramp_samples is not None and ramp_samples > 0:
            ramp_samples = min(ramp_samples, len(audio))
            if is_outro:
                head = audio[:-ramp_samples]
                ramp_region = audio[-ramp_samples:]
            else:
                ramp_region = audio[:ramp_samples]
                tail = audio[ramp_samples:]
            ramped = self._apply_ramp_internal(ramp_region, peak_stretch, is_outro)
            if is_outro:
                return np.concatenate([head, ramped]) if len(ramped) > 0 else head
            else:
                return np.concatenate([ramped, tail]) if len(ramped) > 0 else tail
        else:
            return self._apply_ramp_internal(audio, peak_stretch, is_outro)

    def _apply_ramp_internal(self, audio: np.ndarray,
                             peak_stretch: float,
                             is_outro: bool) -> np.ndarray:
        """Core ramp logic — operates on the ramp window only."""
        try:
            n = len(audio)
            n_segs = max(2, min(12, n // int(2.0 * self.sr)))
            if n // n_segs < 1024:
                return audio

            t = np.linspace(0.0, 1.0, n_segs)
            if is_outro:
                rates = 1.0 + peak_stretch * (1.0 - np.cos(t * np.pi)) * 0.5
            else:
                rates = 1.0 + peak_stretch * (1.0 - np.cos((1.0 - t) * np.pi)) * 0.5

            overlap_ms = 100.0  # ~1/4 beat at 150 BPM — smooths segment boundaries
            overlap = int(overlap_ms * self.sr / 1000.0)

            boundaries = np.linspace(0, n, n_segs + 1, dtype=int)
            out = None

            for i in range(n_segs):
                s, e = int(boundaries[i]), int(boundaries[i + 1])
                if e - s < 512:
                    continue
                seg = audio[s:e]
                stretched = self._pv_stretch(seg, rates[i])

                if out is None:
                    out = stretched
                    continue

                ov = min(overlap, len(out), len(stretched))
                if ov <= 0:
                    out = np.concatenate([out, stretched])
                else:
                    ct = np.linspace(0.0, 1.0, ov, dtype=np.float32)
                    fo = np.cos(ct * np.pi / 2.0)
                    fi = np.sin(ct * np.pi / 2.0)
                    if out.ndim > 1:
                        fo = fo.reshape(-1, 1)
                        fi = fi.reshape(-1, 1)
                    blended = (out[-ov:] * fo) + (stretched[:ov] * fi)
                    out = np.concatenate([out[:-ov], blended, stretched[ov:]])

            if out is None or len(out) == 0:
                return audio

            if len(out) > n:
                fade_n = min(int(0.005 * self.sr), len(out) - n)
                if fade_n > 0:
                    fade = (np.linspace(1, 0, fade_n).reshape(-1, 1)
                            if out.ndim > 1 else np.linspace(1, 0, fade_n))
                    out[n - fade_n:n] *= fade
                out = out[:n]
            return out
        except Exception as e:
            logger.warning(f"  Tempo ramp failed: {e}")
            return audio

    # ── Phase Vocoder (librosa) ─────────────────────────────────────

    @staticmethod
    def _pv_stretch_mono(seg: np.ndarray, rate: float) -> np.ndarray:
        """Pitch-preserving time-stretch for mono audio."""
        seg_f = seg.astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return librosa.effects.time_stretch(seg_f, rate=rate)

    @staticmethod
    def _pv_stretch(seg: np.ndarray, rate: float) -> np.ndarray:
        """Pitch-preserving time-stretch for mono or stereo audio."""
        if seg.ndim == 1:
            return TempoSync._pv_stretch_mono(seg, rate)
        chans = [TempoSync._pv_stretch_mono(seg[:, ch], rate) for ch in range(seg.shape[1])]
        if any(len(c) == 0 for c in chans):
            return seg
        # Phase vocoder may produce slightly different lengths per channel.
        # Pad the shorter channel with its last sample (edge sustain) instead
        # of trimming the longer one — trimming loses timing and can shift
        # the stereo image by a few ms.
        max_len = max(len(c) for c in chans)
        chans = [
            np.pad(c, (0, max_len - len(c)), mode='edge')
            if len(c) < max_len else c
            for c in chans
        ]
        return np.column_stack(chans)
