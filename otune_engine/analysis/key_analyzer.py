"""
Key detection using the Krumhansl-Schmuckler algorithm.

Correlates chroma features against major and minor key profiles
to determine the most likely key and mode with confidence scoring.
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Krumhansl-Schmuckler key profiles (empirical cognitive weights)
MAJOR_PROFILE = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88
])
MINOR_PROFILE = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17
])

KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Pre-rotated key profiles (each normalized): 12 major + 12 minor
MAJOR_ROTATED = np.array([np.roll(MAJOR_PROFILE, k) / np.sum(MAJOR_PROFILE) for k in range(12)])
MINOR_ROTATED = np.array([np.roll(MINOR_PROFILE, k) / np.sum(MINOR_PROFILE) for k in range(12)])


def detect_key(chroma: np.ndarray) -> Tuple[int, str, str, float]:
    """
    Detect musical key from chroma features.

    Args:
        chroma: Chroma feature matrix (12 × time_frames).

    Returns:
        (key_index, key_name, mode, confidence)
        key_index: 0–11 (C through B)
        mode: 'major' or 'minor'
        confidence: 0.0–1.0
    """
    chroma_mean = np.mean(chroma, axis=1)

    total = np.sum(chroma_mean)
    if total <= 0:
        return 0, 'C', 'major', 0.0
    chroma_mean = chroma_mean / total

    best_corr = -1.0
    best_key = 0
    best_mode = 'major'

    for key_idx in range(12):
        # Test major
        corr = np.corrcoef(chroma_mean, MAJOR_ROTATED[key_idx])[0, 1]
        if not np.isfinite(corr):
            corr = 0.0
        if corr > best_corr:
            best_corr = corr
            best_key = key_idx
            best_mode = 'major'

        # Test minor
        corr = np.corrcoef(chroma_mean, MINOR_ROTATED[key_idx])[0, 1]
        if not np.isfinite(corr):
            corr = 0.0
        if corr > best_corr:
            best_corr = corr
            best_key = key_idx
            best_mode = 'minor'

    confidence = float(np.clip((best_corr - 0.3) / 0.6, 0.0, 1.0))
    if not np.isfinite(confidence):
        confidence = 0.0
    key_name = KEY_NAMES[best_key]

    logger.info(f"  Key: {key_name} {best_mode} (confidence: {confidence:.2f})")
    return best_key, key_name, best_mode, confidence
