# O-Tune Engine v2

Apple Music-style automated DJ mixing engine optimized for Apple Silicon. Analyzes tracks for tempo, key, energy, vocals, and structure, then creates seamless transitions with vocal-aware frequency-sweep crossfades.

## Features

- **6 transition styles** — `apple_automix`, `harmonic_layer`, `energy_punch`, `build_drop`, `palate_cleanser`, `smooth_blend`
- **Vocal-aware crossfading** — LR4 Linkwitz-Riley 3-band frequency sweep with ducking preserves vocal clarity during transitions
- **Smart ordering** — mood progression (sad → happy), key compatibility, energy curves
- **Beat-snapped alignment** — phrase → downbeat → beat-phase → micro-correlation alignment hierarchy
- **Tempo ramp** — pitch-preserving phase vocoder for seamless tempo changes during crossfades
- **Apple Silicon GPU acceleration** — Metal Performance Shaders (MPS) for STFT/chroma/cross-correlation
- **Per-pair transition export** — each transition saved as a separate file alongside the full mix
- **EBU R128 loudness normalization** — consistent listening levels across tracks

## Requirements

- Python ≥ 3.10
- macOS (Apple Silicon recommended for GPU acceleration)

### Dependencies

| Package | Minimum | Purpose |
|---|---|---|
| `librosa` | 0.10.0 | Audio analysis, beat tracking, phase vocoder |
| `numpy` | 1.24.0 | Array computation |
| `scipy` | 1.10.0 | Signal processing, filtering, correlation |
| `soundfile` | 0.12.0 | Audio file I/O |
| `pyloudnorm` | 0.1.0 | EBU R128 loudness normalization |
| `psutil` | 5.9.0 | System memory monitoring |
| `torch` | 1.12.0 (optional) | GPU acceleration via MPS |

## Installation

```bash
git clone https://github.com/yourusername/otune-engine.git
cd otune-engine
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

With GPU support:

```bash
pip install -e ".[gpu]"
```

## Usage

### CLI

```bash
# Mix all tracks in a folder
python otune.py /path/to/tracks/

# 30-second crossfade transitions
python otune.py /path/to/tracks/ -c 30

# Custom output name
python otune.py /path/to/tracks/ -o my_mix.wav

# CPU-only mode
python otune.py /path/to/tracks/ --no-gpu

# Start from a specific track (1-based)
python otune.py /path/to/tracks/ --start-track 3

# Vocal ducking strategy
python otune.py /path/to/tracks/ --vocal-mode duck

# Skip interactive prompt
python otune.py /path/to/tracks/ --non-interactive

# Verbose logging
python otune.py /path/to/tracks/ -v
```

### Output

All output goes into `otunedResult/` inside the input folder:

```
tracks/
├── Đôi Mắt Người Xưa.mp3
├── Tâm Sự Đời Tôi.mp3
├── Tình Nhỏ Mau Quên.mp3
├── Anh Là Tia Nắng Trong Em - Lâm Thúy Vân.mp3
├── Vợ Tôi.mp3
└── Gõ Cửa Trái Tim.mp3

otunedResult/
├── otune_mix.wav                              # full mix (28:13)
├── 001_Đôi Mắt Người Xưa__Tâm Sự Đời Tôi.wav   # transition 1  (42s)
├── 002_Tâm Sự Đời Tôi__Tình Nhỏ Mau Quên.wav   # transition 2  (43s)
├── 003_Tình Nhỏ Mau Quên__Anh Là Tia Nắng...wav # transition 3  (23s)
├── 004_Anh Là Tia Nắng...__Vợ Tôi.wav          # transition 4  (29s)
└── 005_Vợ Tôi__Gõ Cửa Trái Tim.wav             # transition 5  (23s)
```

Each transition file is a self-contained clip centered on the crossfade with
10 s of musical context on either side — ready to preview or drop into a
playlist.

### As a library

```python
from otune_engine.core.config import MixConfig
from otune_engine.core.pipeline import MixPipeline
from pathlib import Path

config = MixConfig(
    input_folder=Path("/path/to/tracks"),
    output_file="my_mix.wav",
)
pipeline = MixPipeline(config)
pipeline.run()
```

## How It Works

The pipeline runs in six stages:

1. **Discovery** — scans the input folder for supported audio files (.mp3, .wav, .flac, .m4a, .aac, .ogg)
2. **Selection** — interactive prompt for track order or start point (skippable with `--non-interactive`)
3. **Analysis** — each track is analyzed in parallel for tempo, key, energy, spectral features, vocal segments, structure, genre, and mood
4. **Ordering** — tracks are ordered for optimal flow using mood progression, key compatibility, and energy curves
5. **Mixing** — for each adjacent pair:
   - Transition style is selected based on musical context
   - Crossfade duration is snapped to phrase/downbeat boundaries
   - Beats are aligned at phrase → downbeat → beat → sub-millisecond precision
   - Tempo is ramped (vocal-to-vocal excluded to avoid phase-vocoder artifacts)
   - 3-band LR4 frequency-sweep crossfade is applied with vocal ducking
   - Each transition is exported as a standalone file
6. **Finalization** — peak limiting and loudness normalization, then save the full mix

## Transition Styles

| Style | When Used | Behavior |
|---|---|---|
| `harmonic_layer` | Strong key match (≥0.85) | Gentle 0.5-power curve, high overlap |
| `apple_automix` | Default | 0.5-power curve with LR4 frequency sweep, vocal ducking |
| `energy_punch` | Low → high energy jump | Quick cut with short silence gap |
| `build_drop` | Medium → high energy | Build-down then drop into next track |
| `palate_cleanser` | Key clash (≤0.25) | Full fade-out, silence gap, fade-in |
| `smooth_blend` | Fallback | Equal-power cosine crossfade |

## Project Structure

```
otune-engine/
├── otune.py                          # CLI entry point
├── run_otune.py                      # Minimal example
├── pyproject.toml
│
├── otune_engine/                     # Main engine package
│   ├── core/
│   │   ├── config.py                 # Configuration dataclasses
│   │   ├── types.py                  # Shared data types
│   │   └── pipeline.py               # Main orchestrator
│   │
│   ├── analysis/
│   │   ├── audio_loader.py           # Audio I/O & normalization
│   │   ├── spectral_analyzer.py      # STFT, chroma, MFCC
│   │   ├── beat_analyzer.py          # Tempo, downbeats, time sig
│   │   ├── key_analyzer.py           # Key detection
│   │   ├── vocal_analyzer.py         # Vocal segment detection
│   │   ├── structure_analyzer.py     # Song structure parsing
│   │   ├── genre_analyzer.py         # Genre classification
│   │   └── mood_analyzer.py          # Mood estimation
│   │
│   ├── mixing/
│   │   ├── transition_planner.py     # Style selection & params
│   │   ├── track_ordering.py         # Smart playlist ordering
│   │   ├── crossfader.py             # Crossfade styles
│   │   ├── beat_aligner.py           # Beat alignment
│   │   ├── tempo_sync.py             # Phase vocoder tempo ramp
│   │   ├── vocal_crossfade.py        # LR4 frequency-sweep
│   │   ├── transition_ding.py        # Accent sound
│   │   └── assets/                   # Audio assets
│   │
│   ├── acceleration/
│   │   ├── metal_gpu.py              # MPS GPU acceleration
│   │   └── parallel.py               # Parallel analysis
│   │
│   └── cache/
│       └── analysis_cache.py         # Analysis result caching
│
├── songs/                            # Generated output
│   └── otunedResult/
│       ├── otune_mix.wav             # Full mix (28:13)
│       ├── 001_Đôi Mắt Người Xưa__Tâm Sự Đời Tôi.wav
│       ├── 002_Tâm Sự Đời Tôi__Tình Nhỏ Mau Quên.wav
│       ├── 003_Tình Nhỏ Mau Quên__Anh Là Tia Nắng Trong Em.wav
│       ├── 004_Anh Là Tia Nắng Trong Em__Vợ Tôi.wav
│       └── 005_Vợ Tôi__Gõ Cửa Trái Tim.wav
│
├── src/                              # Legacy v1.x
└── tests/
    ├── test_refactored.py
    ├── test_transition_styles.py
    └── test_gpu_acceleration.py
```

## Configuration

All settings are dataclass-based in `otune_engine/core/config.py`:

- **AnalysisConfig** — sample rate, hop length, FFT size, target LUFS, vocal frequency range
- **TransitionConfig** — crossfade duration (3–30s), vocal mode, duck level, ding, tempo ramp
- **AccelerationConfig** — GPU on/off, batch size, worker threads
- **CacheConfig** — enable/disable, cache directory

## Testing

```bash
pytest tests/
```

GPU tests require PyTorch (`pip install -e ".[gpu]"`).

## License

MIT

---

**Credit:** TicTu Tech  
**Developer:** Thanh Solar NEXT
