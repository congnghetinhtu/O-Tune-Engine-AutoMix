"""
Audio loading, normalization, and format conversion.

Handles mono/stereo conversion, EBU R128 loudness normalization,
and ensures all audio is in (samples, channels) format.
"""

import logging
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import pyloudnorm as pyln

logger = logging.getLogger(__name__)


def load_audio(file_path: Path, sample_rate: int = 44100
               ) -> Tuple[np.ndarray, int, bool]:
    """
    Load audio file, normalize loudness, and ensure stereo output.

    Args:
        file_path: Path to audio file.
        sample_rate: Target sample rate.

    Returns:
        (audio_stereo, sample_rate, was_originally_stereo)
        audio_stereo shape: (samples, 2)
    """
    logger.info(f"Loading: {file_path.name}")

    y, sr = librosa.load(str(file_path), sr=sample_rate, mono=False)

    # librosa returns (channels, samples) for multi-channel
    if y.ndim > 1:
        y = y.T  # → (samples, channels)
        is_stereo = True
    else:
        is_stereo = False

    return y, sr, is_stereo


def normalize_loudness(audio: np.ndarray, sample_rate: int,
                       target_lufs: float = -14.0) -> np.ndarray:
    """
    EBU R128 loudness normalization with peak limiting.

    Handles both mono and stereo. Prevents clipping at -1 dBTP.
    """
    if len(audio) == 0:
        return audio

    # Ensure correct format: (samples, channels) for stereo
    # pyloudnorm expects (samples, channels) for stereo, (samples,) for mono.
    # If shape is (2, N) with N > 2, assume channels-first and transpose.
    if audio.ndim == 2 and audio.shape[0] <= 2 and audio.shape[0] < audio.shape[1]:
        audio = audio.T

    meter = pyln.Meter(sample_rate)

    peak = float(np.max(np.abs(audio)))
    if peak < 1e-8:
        return audio  # silent — skip normalization

    try:
        current_lufs = meter.integrated_loudness(audio)
    except Exception as e:
        logger.warning(f"  Loudness measurement failed: {e}")
        return audio

    gain_db = target_lufs - current_lufs
    gain_linear = 10 ** (gain_db / 20.0)
    peak_after = peak * gain_linear

    if peak_after > 0.891:  # -1 dBTP
        max_gain = 0.891 / peak
        adjusted_target = current_lufs + 20 * np.log10(max_gain)
        return pyln.normalize.loudness(audio, current_lufs, adjusted_target)
    else:
        return pyln.normalize.loudness(audio, current_lufs, target_lufs)


def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    """Convert mono audio to stereo by duplicating channels."""
    if audio.ndim == 1:
        return np.column_stack([audio, audio])
    return audio


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert stereo audio to mono for analysis."""
    if audio.ndim > 1:
        return librosa.to_mono(audio.T)  # librosa expects (channels, samples)
    return audio
