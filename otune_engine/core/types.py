"""
Shared type definitions for OTune Engine v2.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class VocalSegment:
    """A detected vocal region within a track."""
    start: float  # seconds
    end: float    # seconds
    energy: float = 0.0  # RMS energy of vocal band in this segment
    confidence: float = 0.5

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class StructureSection:
    """A labeled section of a track (intro/verse/chorus/bridge/outro)."""
    start: float
    end: float
    label: str  # 'intro', 'verse', 'chorus', 'bridge', 'outro', 'instrumental'
    energy: float = 0.0
    vocal_ratio: float = 0.0
    repetition_score: float = 0.0
    boring_score: float = 0.0  # 0–1, higher = more boring (for intros)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TrackAnalysis:
    """Complete analysis result for a single audio track."""
    # Identity
    file_path: Path = field(default_factory=lambda: Path('.'))

    # Audio data (not cached — loaded fresh each time)
    audio_data: Optional[np.ndarray] = field(default=None, repr=False)
    sample_rate: int = 44100
    duration: float = 0.0
    is_stereo: bool = True

    # Tempo and rhythm
    tempo: float = 120.0
    actual_tempo: float = 120.0
    tempo_multiplier: str = 'normal'  # 'normal', 'half-time', 'double-time'
    beats: np.ndarray = field(default_factory=lambda: np.array([]))
    beat_frames: np.ndarray = field(default_factory=lambda: np.array([]))
    downbeats: np.ndarray = field(default_factory=lambda: np.array([]))
    beat_strengths: np.ndarray = field(default_factory=lambda: np.array([]))
    beat_confidence: np.ndarray = field(default_factory=lambda: np.array([]))
    time_signature: int = 4
    swing_ratio: float = 0.5
    groove_type: str = 'straight'  # 'straight', 'swing', 'shuffle'

    # Key and harmony
    key: int = 0  # 0–11 (C, C#, D, ...)
    key_name: str = 'C'
    key_mode: str = 'major'
    key_confidence: float = 0.0

    # Energy and spectral
    energy: float = 0.0
    energy_variation: float = 0.0
    spectral_centroid: float = 0.0
    spectral_rolloff: float = 0.0
    spectral_bandwidth: float = 0.0
    zcr: float = 0.0
    mfcc_mean: np.ndarray = field(default_factory=lambda: np.zeros(13))

    # Vocals
    vocal_segments: List[VocalSegment] = field(default_factory=list)
    has_vocals: bool = False

    # Structure
    structure_sections: List[StructureSection] = field(default_factory=list)
    main_section: Optional[StructureSection] = None
    phrases: List[Tuple[float, float, int]] = field(default_factory=list)
    intro_end: float = 0.0
    outro_start: float = 0.0

    # Genre
    genre_hint: str = 'unknown'

    # Mood (0=sad, 1=happy)
    mood_score: float = 0.5
    mood_label: str = 'neutral'

    # Loudness
    peak_level: float = 0.0
    rms_level: float = 0.0


@dataclass
class TransitionPlan:
    """Plan for transitioning between two tracks."""
    style: str = 'apple_automix'  # 'smooth_blend', 'apple_automix', 'energy_punch', 'harmonic_layer', etc.
    crossfade_duration: float = 8.0
    crossfade_start: float = 0.0  # seconds into track1 where crossfade begins
    intro_skip: float = 0.0  # seconds of track2 intro to skip
    fade_curve_power: float = 1.0
    gap_duration: float = 0.0  # for palate_cleanser / energy_punch
    overlap_boost: float = 0.5
    confidence: float = 0.0
    reason: str = ''

    # Metadata
    section1_label: str = ''  # e.g., 'outro'
    section2_label: str = ''  # e.g., 'intro', 'verse'
    compatibility_score: float = 0.0
