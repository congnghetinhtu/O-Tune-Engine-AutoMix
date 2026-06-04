"""
Configuration dataclasses for OTune Engine v2.
 
All settings are centralized here with sensible defaults.
No more 15-parameter constructors scattered across the codebase.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


@dataclass
class AnalysisConfig:
    """Settings for audio analysis."""
    sample_rate: int = 44100
    hop_length: int = 512
    n_fft: int = 2048
    target_lufs: float = -14.0
    max_analysis_duration: float = 120.0  # seconds — cap vocal/genre analysis length
    vocal_min_segment_duration: float = 1.0  # seconds — discard short vocal blips
    vocal_freq_low: float = 300.0   # Hz — vocal band lower bound
    vocal_freq_high: float = 4000.0  # Hz — vocal band upper bound


@dataclass
class TransitionConfig:
    """Settings for crossfade and transition behavior."""
    crossfade_duration: float = 8.0  # seconds — default crossfade length
    min_crossfade: float = 3.0
    max_crossfade: float = 30.0
    vocal_crossfade_mode: str = 'auto'  # 'auto', 'duck', 'blend', 'gap_align'
    vocal_preserve_both: bool = False  # set True to force blend and keep both vocals
    vocal_duck_db: float = -8.0  # dB attenuation for outgoing vocal during overlap
    ding_enabled: bool = True  # play transition accent sound at blend center
    ding_level_db: float = -2.0  # dB level for transition accent (-6 = clearly audible)
    tempo_ramp_max_pct: float = 2.0  # max tempo change during crossfade (%)
    scratch_bend_enabled: bool = True
    scratch_bend_max_pct: float = 6.0  # max instantaneous speed bend during overlap (%)
    beat_align_window_ms: float = 25.0  # ms — default search window for beat alignment


@dataclass
class AccelerationConfig:
    """Settings for GPU and parallel processing."""
    use_gpu: bool = True
    gpu_batch_size: Optional[int] = None  # None = auto-detect
    max_workers: int = 3  # parallel analysis workers
    use_process_pool: bool = True  # True = ProcessPoolExecutor (GIL-free)


@dataclass
class CacheConfig:
    """Settings for analysis result caching."""
    enabled: bool = True
    cache_dir: Optional[Path] = None  # None = .otune_cache in input folder
    cache_version: str = '2.0'


@dataclass
class MixConfig:
    """Top-level configuration combining all sub-configs."""
    input_folder: Path = field(default_factory=lambda: Path('.'))
    output_file: str = 'otune_mix.wav'
    start_track_index: Optional[int] = None
    custom_order: Optional[List[int]] = None  # 1-based track indices
    non_interactive: bool = False

    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    acceleration: AccelerationConfig = field(default_factory=AccelerationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    # Computed properties
    @property
    def supported_formats(self) -> set:
        return {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'}

    @property
    def cache_path(self) -> Path:
        if self.cache.cache_dir:
            return self.cache.cache_dir
        return self.input_folder / '.otune_cache'
