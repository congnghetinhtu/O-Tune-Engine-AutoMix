"""
Transition accent sound — a satisfying "ding" that plays at the blend
center of each transition, landing on the beat for a polished DJ feel.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import librosa

logger = logging.getLogger(__name__)

DING_PATH = Path(__file__).parent / "assets" / "transition_ding.wav"


class TransitionDing:
    """Loads and places a transition accent sound at the correct beat position."""

    def __init__(self, sample_rate: int = 44100,
                 level_db: float = -6.0,
                 sound_path: Optional[Path] = None):
        self.sr = sample_rate
        self.level_db = level_db
        self.gain = 10 ** (level_db / 20.0)

        path = sound_path or DING_PATH
        if not path.exists():
            logger.warning(f"Transition ding not found: {path}")
            self.audio: Optional[np.ndarray] = None
            return

        audio, sr = librosa.load(str(path), sr=sample_rate, mono=False)
        self.audio = np.ascontiguousarray(audio.T, dtype=np.float32)
        logger.info(
            f"Loaded transition ding: {path.name} "
            f"({len(self.audio)} samples, {level_db:+.0f} dB)"
        )

    def place(self, mix: np.ndarray, sample_pos: int) -> np.ndarray:
        """Mix the ding into *mix* starting at *sample_pos*."""
        if self.audio is None:
            return mix

        n_ding = len(self.audio)
        end = min(sample_pos + n_ding, len(mix))
        n_copy = end - sample_pos
        if n_copy <= 0:
            return mix

        mix[sample_pos:end] += self.audio[:n_copy] * self.gain

        peaks = np.max(np.abs(mix), axis=0) if mix.ndim > 1 else np.max(np.abs(mix))
        if np.any(peaks > 0.99):
            peak_val = float(np.max(peaks))
            scale = 0.95 / peak_val
            mix *= scale
            logger.debug(f"  Ding clamped peaks from {peak_val:.2f} to {mix.shape}")

        return mix
