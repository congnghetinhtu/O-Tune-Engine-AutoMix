"""
Robust vocal detection using energy-percentile thresholding.

Simplified over the previous multi-feature Z-norm approach:
  - Directly thresholds the vocal-band (300–4000 Hz) RMS energy against
    the 25th percentile of non-silent frames. This is more robust than
    Z-norm multi-feature scoring because it adapts to the track's own
    dynamic range and doesn't fail on densely-mixed material.
  - Removed the Z-norm / multi-feature probability: spectral centroid,
    MFCC variance, and chroma strength added complexity but little
    discriminative value compared to plain vocal-band energy.
  - Uses causal sosfilt instead of acausal filtfilt to avoid pre-ringing.
"""

import logging
from typing import List, Tuple

import librosa
import numpy as np
from scipy import signal as sig

from ..core.types import VocalSegment

logger = logging.getLogger(__name__)

_VOCAL_BAND_RMS_PERCENTILE = 25.0  # frames above this percentile → vocal


class VocalAnalyzer:
    """Detect vocal segments via energy-percentile thresholding."""

    def __init__(self, sample_rate: int = 44100,
                 freq_low: float = 300.0, freq_high: float = 4000.0,
                 min_segment_duration: float = 1.0,
                 max_analysis_seconds: float = 120.0):
        self.sr = sample_rate
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.min_duration = min_segment_duration
        self.max_samples = int(max_analysis_seconds * sample_rate)

        # Causal 4th-order Butterworth bandpass (SOS format)
        nyquist = sample_rate / 2.0
        low = freq_low / nyquist
        high = min(freq_high / nyquist, 0.99)
        self._bp_sos = sig.butter(4, [low, high], btype='band', output='sos')

        # Spectral centroid computed during analyze, passed to _detect_segments

    def analyze(self, y_mono: np.ndarray) -> dict:
        """Vocal analysis using energy-percentile thresholding."""
        y_short = y_mono[:min(len(y_mono), self.max_samples)]

        # HPSS to isolate harmonic content
        y_harmonic, _ = librosa.effects.hpss(y_short, margin=2.0)

        # Bandpass to vocal frequency range (causal sosfilt, no pre-ringing)
        vocal_band = sig.sosfilt(self._bp_sos, y_harmonic).astype(np.float32)

        hop = 1024

        # Compute spectral centroid from harmonic signal for vocal refinement.
        # Vocals typically have centroid 500-2000 Hz; instruments in the same
        # band (guitar, brass) tend to be brighter or darker.
        centroid = librosa.feature.spectral_centroid(
            y=y_harmonic, sr=self.sr, hop_length=hop
        )[0]

        segments, frame_times = self._detect_segments(vocal_band, hop, centroid)
        energy_env = self._vocal_energy_envelope(vocal_band, hop)
        breath_gaps = self._detect_breath_gaps(segments)

        has_vocals = len(segments) > 0
        if has_vocals:
            total_vocal = sum(s.duration for s in segments)
            logger.info(f"  {len(segments)} vocal segments, "
                        f"{total_vocal:.1f}s total vocal content")
        else:
            logger.info(f"  No vocals detected (instrumental track)")

        return {
            'vocal_segments': segments,
            'has_vocals': has_vocals,
            'vocal_energy_envelope': energy_env,
            'breath_gaps': breath_gaps,
        }

    # ── Segment Detection ───────────────────────────────────────────

    def _detect_segments(self, vocal_band: np.ndarray, hop: int,
                         centroid: np.ndarray = None
                         ) -> Tuple[List[VocalSegment], np.ndarray]:
        """Detect vocal segments using percentile-thresholded RMS energy
        refined by spectral centroid to reduce false positives from
        non-vocal instruments in the same frequency range (e.g., electric
        guitar, brass)."""

        vocal_rms = librosa.feature.rms(y=vocal_band, hop_length=hop)[0]

        frame_times = librosa.frames_to_time(
            np.arange(len(vocal_rms)), sr=self.sr, hop_length=hop
        )

        # Adaptive threshold: percentile of non-silent frames
        floor = 1e-8
        non_silent = vocal_rms[vocal_rms > floor]
        if len(non_silent) == 0:
            return [], frame_times
        threshold = float(np.percentile(non_silent, _VOCAL_BAND_RMS_PERCENTILE))

        is_vocal = vocal_rms > threshold

        # Refine: reject frames where spectral centroid is far from typical
        # vocal range (~300-3000 Hz). Instruments like electric guitar,
        # overdriven synths, or brass in the 300-4000 Hz band have
        # significantly higher centroids than vocals.
        if centroid is not None:
            centroid_too_high = centroid > 3500.0
            centroid_too_low = centroid < 200.0
            is_vocal = is_vocal & (~centroid_too_high) & (~centroid_too_low)

        segments = []
        in_vocal = False
        start_time = 0.0

        for i, v in enumerate(is_vocal):
            t = frame_times[i] if i < len(frame_times) else frame_times[-1]
            if v and not in_vocal:
                start_time = t
                in_vocal = True
            elif not v and in_vocal:
                if t - start_time > self.min_duration:
                    seg_energy = float(np.mean(
                        vocal_rms[max(0, int(start_time * self.sr // hop)):
                                  min(len(vocal_rms), int(t * self.sr // hop))]
                    ))
                    segments.append(VocalSegment(
                        start=start_time, end=t,
                        energy=seg_energy,
                        confidence=np.clip(seg_energy / max(non_silent), 0, 1),
                    ))
                in_vocal = False

        # Handle segment ending at track boundary
        if in_vocal and len(frame_times) > 0:
            end_t = frame_times[-1]
            if end_t - start_time > self.min_duration:
                seg_energy = float(np.mean(
                    vocal_rms[max(0, int(start_time * self.sr // hop)):
                              min(len(vocal_rms), int(end_t * self.sr // hop))]
                ))
                segments.append(VocalSegment(
                    start=start_time, end=end_t,
                    energy=seg_energy,
                    confidence=np.clip(seg_energy / max(non_silent), 0, 1),
                ))

        return segments, frame_times

    # ── Energy Envelope ─────────────────────────────────────────────

    def _vocal_energy_envelope(self, vocal_band: np.ndarray,
                               hop: int) -> np.ndarray:
        """Compute per-frame RMS energy of the vocal frequency band."""
        rms = librosa.feature.rms(y=vocal_band, hop_length=hop)[0]
        # Smooth for cleaner envelope
        if len(rms) > 5:
            from scipy.ndimage import gaussian_filter1d
            rms = gaussian_filter1d(rms, sigma=2.0)
        return rms

    # ── Breath Gap Detection ────────────────────────────────────────

    def _detect_breath_gaps(self, segments: List[VocalSegment]) -> List[Tuple[float, float]]:
        """
        Find gaps between vocal segments (breath points).
        These are ideal anchor points for vocal-to-vocal transitions.
        """
        if len(segments) < 2:
            return []

        gaps = []
        for i in range(len(segments) - 1):
            gap_start = segments[i].end
            gap_end = segments[i + 1].start
            gap_duration = gap_end - gap_start

            # Keep gaps between 0.3s and 5.0s (natural breath/phrase pauses)
            if 0.3 <= gap_duration <= 5.0:
                gaps.append((gap_start, gap_end))

        logger.info(f"  {len(gaps)} vocal breath gaps detected")
        return gaps
