"""
Spectral feature extraction.

Computes STFT (once, reused), spectral centroid, rolloff, bandwidth,
MFCC, ZCR, and chroma features. Uses GPU when available.
"""

import logging
from typing import Optional, Tuple

import librosa
import numpy as np

logger = logging.getLogger(__name__)


class SpectralAnalyzer:
    """Compute and cache spectral features from audio."""

    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048,
                 hop_length: int = 512, gpu=None):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.gpu = gpu  # Optional MetalGPU instance

    def compute_all(self, y_mono: np.ndarray) -> dict:
        """
        Compute all spectral features in one pass.

        Returns dict with keys: stft, chroma, rms, energy, energy_variation,
        spectral_centroid, spectral_rolloff, spectral_bandwidth, zcr, mfcc_mean
        """
        # STFT — compute once, reuse everywhere
        if self.gpu and self.gpu.use_mps:
            S = self.gpu.compute_stft(y_mono, self.n_fft, self.hop_length)
        else:
            S = np.abs(librosa.stft(y_mono, n_fft=self.n_fft,
                                     hop_length=self.hop_length))

        # Chroma
        if self.gpu and self.gpu.use_mps:
            chroma = self.gpu.compute_chroma(S, self.sr)
        else:
            chroma = librosa.feature.chroma_stft(S=S, sr=self.sr)

        # RMS energy (from pre-computed STFT — 2x faster than recomputing)
        rms = librosa.feature.rms(S=S, frame_length=self.n_fft,
                                   hop_length=self.hop_length)[0]

        # Spectral features (all from pre-computed STFT)
        spectral_centroid = float(np.mean(
            librosa.feature.spectral_centroid(S=S, sr=self.sr)))
        spectral_rolloff = float(np.mean(
            librosa.feature.spectral_rolloff(S=S, sr=self.sr)))
        spectral_bandwidth = float(np.mean(
            librosa.feature.spectral_bandwidth(S=S, sr=self.sr)))

        # Zero crossing rate
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y_mono)))

        # MFCC (13 coefficients) — reuse pre-computed STFT to avoid redundant FFT
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S ** 2), sr=self.sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)

        return {
            'stft': S,
            'chroma': chroma,
            'mfccs': mfccs,
            'rms': rms,
            'energy': float(np.mean(rms)),
            'energy_variation': float(np.std(rms)),
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'spectral_bandwidth': spectral_bandwidth,
            'zcr': zcr,
            'mfcc_mean': mfcc_mean,
        }
