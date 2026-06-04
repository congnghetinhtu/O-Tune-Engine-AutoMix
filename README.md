# 🎧 O-Tune Engine v2

<p align="center">
  <img src="https://img.shields.io/badge/python-≥3.10-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/macOS-Apple_Silicon-000000?style=for-the-badge&logo=apple&logoColor=white"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/version-2.0.0-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/dj-Automix-FF6C37?style=for-the-badge&logo=beatport&logoColor=white"/>
</p>

<p align="center">
  <b>Apple Music‑style automated DJ mixing engine</b><br>
  Analyzes tracks for tempo, key, energy, vocals & structure —<br>
  then creates seamless transitions with vocal‑aware frequency‑sweep crossfades.
</p>

---

## ✨ Features

| | |
|---|---|
| 🎛️ **6 transition styles** | `apple_automix`, `harmonic_layer`, `energy_punch`, `build_drop`, `palate_cleanser`, `smooth_blend` |
| 🎤 **Vocal‑aware crossfading** | LR4 Linkwitz‑Riley 3‑band frequency sweep with ducking preserves vocal clarity |
| 🧠 **Smart ordering** | Mood progression (sad → happy), key compatibility, energy curves |
| 🥁 **Beat‑snapped alignment** | Phrase → downbeat → beat‑phase → micro‑correlation hierarchy |
| ⏱️ **Tempo ramp** | Pitch‑preserving phase vocoder for seamless BPM changes |
| 🚀 **Apple Silicon GPU** | Metal Performance Shaders (MPS) for STFT / chroma / cross‑correlation |
| 📦 **Per‑pair export** | Each transition saved as a standalone file + full mix |
| 🔊 **EBU R128 loudness** | Consistent listening levels across every track |

---

## 📋 Requirements

- Python **≥ 3.10**
- **macOS** (Apple Silicon recommended for GPU)

### Dependencies

| Package | Min | Role |
|---|---|---|
| `librosa` | 0.10.0 | 🎵 Audio analysis, beat tracking, phase vocoder |
| `numpy` | 1.24.0 | 🔢 Array computation |
| `scipy` | 1.10.0 | ⚙️ Signal processing, filtering, correlation |
| `soundfile` | 0.12.0 | 💾 Audio file I/O |
| `pyloudnorm` | 0.1.0 | 📊 EBU R128 loudness normalisation |
| `psutil` | 5.9.0 | 📈 System memory monitor |
| `torch` | 1.12.0 *opt.* | 🚀 GPU acceleration via MPS |

---

## 🛠️ Installation

```bash
git clone https://github.com/congnghetinhtu/O-Tune-Engine-AutoMix.git
cd O-Tune-Engine-AutoMix
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

With GPU support:

```bash
pip install -e ".[gpu]"
```

---

## 🚀 Usage

### CLI

```bash
# Mix all tracks in a folder
python otune.py /path/to/tracks/

# 30‑second crossfade transitions
python otune.py /path/to/tracks/ -c 30

# Custom output name
python otune.py /path/to/tracks/ -o my_mix.wav

# CPU‑only mode
python otune.py /path/to/tracks/ --no-gpu

# Start from a specific track (1‑based)
python otune.py /path/to/tracks/ --start-track 3

# Vocal ducking strategy
python otune.py /path/to/tracks/ --vocal-mode duck

# Skip interactive prompt
python otune.py /path/to/tracks/ --non-interactive

# Verbose logging
python otune.py /path/to/tracks/ -v
```

### 📁 Output

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
├── otune_mix.wav                              ⬅️ full mix (28:13)
├── 001_Đôi Mắt Người Xưa__Tâm Sự Đời Tôi.wav   ⬅️ transition 1  (42s)
├── 002_Tâm Sự Đời Tôi__Tình Nhỏ Mau Quên.wav   ⬅️ transition 2  (43s)
├── 003_Tình Nhỏ Mau Quên__Anh Là Tia Nắng...wav ⬅️ transition 3  (23s)
├── 004_Anh Là Tia Nắng...__Vợ Tôi.wav          ⬅️ transition 4  (29s)
└── 005_Vợ Tôi__Gõ Cửa Trái Tim.wav             ⬅️ transition 5  (23s)
```

Each transition is a self‑contained clip with **10 s of musical context** on either side of the
crossfade — ready to preview or drop straight into a playlist.

### 📦 As a library

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

---

## ⚙️ How It Works

```
Discovery  →  Selection  →  Analysis  →  Ordering  →  Mixing  →  Finalization
```

| # | Stage | What happens |
|---|---|---|
| 1 | 🔍 **Discovery** | Scans folder for `.mp3` / `.wav` / `.flac` / `.m4a` / `.aac` / `.ogg` |
| 2 | 🎯 **Selection** | Interactive prompt for track order or start point |
| 3 | 📊 **Analysis** | Parallel analysis of tempo, key, energy, vocals, structure, genre, mood |
| 4 | 🧩 **Ordering** | Smart playlist with mood progression + key/energy compatibility |
| 5 | 🎚️ **Mixing** | Style selection → phrase‑snapped duration → hierarchical beat alignment → tempo ramp → 3‑band LR4 frequency‑sweep crossfade with vocal ducking → per‑pair export |
| 6 | 🎵 **Finalization** | Peak limiting + loudness normalization → save full mix |

---

## 🎨 Transition Styles

| Style | Trigger | Behaviour |
|---|---|---|
| `harmonic_layer` | Strong key match (≥0.85) | Gentle 0.5‑power curve, high overlap |
| `apple_automix` | ⭐ Default | 0.5‑power curve + LR4 sweep + vocal ducking |
| `energy_punch` | Low → high energy jump | Quick cut with short silence gap |
| `build_drop` | Medium → high energy | Build‑down then drop into next track |
| `palate_cleanser` | Key clash (≤0.25) | Full fade‑out → silence → fade‑in |
| `smooth_blend` | Fallback | Equal‑power cosine crossfade |

---

## 📁 Project Structure

```
otune-engine/
├── 🚪 otune.py                       # CLI entry point
├── 📄 run_otune.py                   # Minimal example
├── 📦 pyproject.toml
│
├── 🧠 otune_engine/                  # Main engine package
│   ├── core/                         #   Orchestration
│   │   ├── config.py                 #     Dataclass configs
│   │   ├── types.py                  #     Shared types
│   │   └── pipeline.py               #     Main orchestrator
│   │
│   ├── analysis/                     #   Feature extraction
│   │   ├── audio_loader.py           #     I/O & loudness
│   │   ├── spectral_analyzer.py      #     STFT, chroma, MFCC
│   │   ├── beat_analyzer.py          #     Tempo, downbeats
│   │   ├── key_analyzer.py           #     Key detection
│   │   ├── vocal_analyzer.py         #     Vocal segments
│   │   ├── structure_analyzer.py     #     Song structure
│   │   ├── genre_analyzer.py         #     Genre (17 types)
│   │   └── mood_analyzer.py          #     Mood estimation
│   │
│   ├── mixing/                       #   Transition engine
│   │   ├── transition_planner.py     #     Style & params
│   │   ├── track_ordering.py         #     Smart ordering
│   │   ├── crossfader.py             #     6 crossfade styles
│   │   ├── beat_aligner.py           #     Beat alignment
│   │   ├── tempo_sync.py             #     Phase vocoder
│   │   ├── vocal_crossfade.py        #     LR4 frequency sweep
│   │   ├── transition_ding.py        #     Accent sound
│   │   └── assets/                   #     Audio assets
│   │
│   ├── acceleration/                 #   Performance
│   │   ├── metal_gpu.py              #     MPS GPU
│   │   └── parallel.py               #     Parallel analysis
│   │
│   └── cache/
│       └── analysis_cache.py         #   JSON cache
│
├── 💿 songs/otunedResult/            # Generated output
│   ├── otune_mix.wav
│   ├── 001_*__*.wav
│   └── ...
│
├── 📜 src/                           # Legacy v1.x
└── 🧪 tests/
    ├── test_refactored.py
    ├── test_transition_styles.py
    └── test_gpu_acceleration.py
```

---

## ⚙️ Configuration

All settings live in `otune_engine/core/config.py`:

| Config | What you can tweak |
|---|---|
| `AnalysisConfig` | Sample rate, hop length, FFT size, target LUFS, vocal frequency range |
| `TransitionConfig` | Crossfade duration (3–30 s), vocal mode, duck level, ding, tempo ramp % |
| `AccelerationConfig` | GPU on/off, batch size, worker threads |
| `CacheConfig` | Enable/disable, cache directory |

---

## 🧪 Testing

```bash
pytest tests/
```

GPU tests require PyTorch (`pip install -e ".[gpu]"`).

---

## 📄 License

**MIT** — Free, permissive, open‑source.

You can use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of this software. The only requirement is that you include the original copyright notice and permission notice in all copies or substantial portions of the software.

**In plain language:** Do whatever you want — just keep the credit notice.

---

<p align="center">
  <b>Credit:</b> TicTu Tech &nbsp;·&nbsp; <b>Developer:</b> Thanh Solar NEXT
</p>
