# 🎧 O-Tune Engine v2

<p align="center">
  <img src="https://img.shields.io/badge/python-≥3.10-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/macOS-Apple_Silicon-000000?style=for-the-badge&logo=apple&logoColor=white"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/version-2.0.0-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/dj-Automix-FF6C37?style=for-the-badge&logo=beatport&logoColor=white"/>
</p>

<p align="center">
  <b>An Apple Music‑style automated DJ mixing engine for your own music.</b><br>
  Throw in a folder of songs — it figures out tempo, key, energy, vocals, and structure,<br>
  then weaves them together with seamless, vocal‑aware crossfades.
</p>

---

## 🥊 How O‑Tune Stacks Up Against Apple Music AutoMix

Apple Music's **AutoMix** (or Crossfade, or whatever they call it in your region) is basically a dimmer switch between songs. It fades one out while fading the next in — no analysis, no beat-matching, no nothing. It works, but it's blind.

**O‑Tune is the opposite of blind.** It listens hard:

| Feature | 🍎 Apple Music AutoMix | 🎧 O‑Tune Engine |
|---|---|---|
| **Tempo detection** | ❌ Nope | ✅ Catches half-time, double-time, even swing |
| **Tempo ramping** | ❌ Abrupt jump | ✅ Pitch‑preserving phase vocoder — no chipmunk effect |
| **Beat alignment** | ❌ Ignored | ✅ Snaps to phrase → downbeat → beat → micro‑sample |
| **Key matching** | ❌ Doesn't care | ✅ Krumhansl‑Schmuckler + compatibility scoring |
| **Vocal awareness** | ❌ Vocals clash all the time | ✅ Splits lows/mids/highs, ducks only what needs ducking |
| **Energy analysis** | ❌ None | ✅ Spectral energy + variation + mood estimation |
| **Transition styles** | ❌ One boring fade | ✅ 6 different styles, chosen per transition |
| **Smart ordering** | ❌ Album order or bust | ✅ Mood curve + harmonic compatibility |
| **GPU acceleration** | ❌ N/A | ✅ Apple Silicon MPS — M1 through M4 |
| **Per‑pair export** | ❌ One mix blob | ✅ Every transition saved as its own file |
| **Loudness** | ❌ Raw levels | ✅ EBU R128 (−14 LUFS) — consistent volume everywhere |
| **Genre detection** | ❌ None | ✅ 17 genre tags |

**In short:** Apple Music AutoMix is a crossfader. O‑Tune Engine is a DJ who actually preps the set.

---

## ✨ What Makes It Special

| | |
|---|---|
| 🎛️ **6 transition styles** | `apple_automix`, `harmonic_layer`, `energy_punch`, `build_drop`, `palate_cleanser`, `smooth_blend` |
| 🎤 **Vocal‑aware crossfading** | A 3‑band frequency sweep with ducking so vocals never get buried |
| 🧠 **Smart ordering** | Goes from sad → happy, keeps keys and energy curves smooth |
| 🥁 **Beat‑snapped alignment** | Locks onto phrases, downbeats, beats — down to the sample |
| ⏱️ **Tempo ramp** | Changes BPM without changing pitch (phase vocoder magic) |
| 🚀 **Apple Silicon GPU** | Uses MPS for STFT, chroma, cross‑correlation — fast |
| 📦 **Per‑pair export** | Every transition is a standalone file + the full mix |
| 🔊 **EBU R128 loudness** | So you're not reaching for the volume knob every track |

---

## 📋 What You'll Need

- Python **3.10 or newer**
- A **Mac** — Apple Silicon is nice for GPU, but Intel works too

### Dependencies

| Package | Min version | What it's for |
|---|---|---|
| `librosa` | 0.10.0 | 🎵 Audio analysis, beat tracking, the phase vocoder |
| `numpy` | 1.24.0 | 🔢 Everything numerical |
| `scipy` | 1.10.0 | ⚙️ Filtering, correlation, signal processing |
| `soundfile` | 0.12.0 | 💾 Reading and writing audio files |
| `pyloudnorm` | 0.1.0 | 📊 Making everything sound equally loud |
| `psutil` | 5.9.0 | 📈 Keeping an eye on memory usage |
| `torch` | 1.12.0 *optional* | 🚀 GPU acceleration via MPS |

---

## 🛠️ Getting It Running

```bash
git clone https://github.com/congnghetinhtu/O-Tune-Engine-AutoMix.git
cd O-Tune-Engine-AutoMix
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Want GPU?

```bash
pip install -e ".[gpu]"
```

---

## 🚀 Using It

### From the terminal

```bash
# Mix everything in a folder
python otune.py /path/to/tracks/

# 30‑second crossfades
python otune.py /path/to/tracks/ -c 30

# Name your mix
python otune.py /path/to/tracks/ -o my_mix.wav

# CPU only (no GPU)
python otune.py /path/to/tracks/ --no-gpu

# Start at a specific track (1‑based)
python otune.py /path/to/tracks/ --start-track 3

# Vocal ducking
python otune.py /path/to/tracks/ --vocal-mode duck

# Skip the "which track first?" prompt
python otune.py /path/to/tracks/ --non-interactive

# See what's happening under the hood
python otune.py /path/to/tracks/ -v
```

### 📁 What comes out

Everything lands in `otunedResult/` inside your input folder:

```
tracks/
├── Đôi Mắt Người Xưa.mp3
├── Tâm Sự Đời Tôi.mp3
├── Tình Nhỏ Mau Quên.mp3
├── Anh Là Tia Nắng Trong Em - Lâm Thúy Vân.mp3
├── Vợ Tôi.mp3
└── Gõ Cửa Trái Tim.mp3

otunedResult/
├── otune_mix.wav                              ⬅️ Full mix (28:13)
├── 001_Đôi Mắt Người Xưa__Tâm Sự Đời Tôi.wav   ⬅️ Transition 1  (42s)
├── 002_Tâm Sự Đời Tôi__Tình Nhỏ Mau Quên.wav   ⬅️ Transition 2  (43s)
├── 003_Tình Nhỏ Mau Quên__Anh Là Tia Nắng...wav ⬅️ Transition 3  (23s)
├── 004_Anh Là Tia Nắng...__Vợ Tôi.wav          ⬅️ Transition 4  (29s)
└── 005_Vợ Tôi__Gõ Cửa Trái Tim.wav             ⬅️ Transition 5  (23s)
```

Each transition clip gives you about **10 seconds of musical runway** on either side of the crossfade — ready to preview or drop straight into a playlist.

### 📦 Using it as a library

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

## ⚙️ How It Works (the short version)

```
Discovery  →  Selection  →  Analysis  →  Ordering  →  Mixing  →  Finalization
```

| # | Stage | What's happening |
|---|---|---|
| 1 | 🔍 **Discovery** | Scans your folder for `.mp3`, `.wav`, `.flac`, `.m4a`, `.aac`, `.ogg` |
| 2 | 🎯 **Selection** | Asks you which track to start with, or just rolls through them |
| 3 | 📊 **Analysis** | Runs everything in parallel — tempo, key, energy, vocals, structure, genre, mood |
| 4 | 🧩 **Ordering** | Builds a smart playlist: mood curves + harmonic compatibility |
| 5 | 🎚️ **Mixing** | Picks a style → aligns to phrase/downbeat/beat → ramps tempo → 3‑band frequency‑sweep crossfade → saves the pair |
| 6 | 🎵 **Finalization** | Adds a peak limiter, evens out loudness, writes the full mix |

---

## 🎨 The 6 Transition Styles

| Style | When it kicks in | How it sounds |
|---|---|---|
| `harmonic_layer` | Strong key match (≥0.85) | Gentle 0.5‑power curve, lots of overlap — buttery |
| `apple_automix` | ⭐ Default | 0.5‑power curve + LR4 sweep + vocal ducking — the all‑rounder |
| `energy_punch` | Low → high energy jump | Quick cut with a tiny silence gap — wakes you up |
| `build_drop` | Medium → high energy | A mini build‑down, then drops into the next track |
| `palate_cleanser` | Key clash (≤0.25) | Full fade‑out → silence → fade‑in — reset button |
| `smooth_blend` | Fallback | Equal‑power cosine crossfade — simple and reliable |

---

## 📁 Project Structure

```
otune-engine/
├── 🚪 otune.py                       # CLI — the front door
├── 📄 run_otune.py                   # Minimal example script
├── 📦 pyproject.toml
│
├── 🧠 otune_engine/                  # The engine room
│   ├── core/                         #   Orchestrating everything
│   │   ├── config.py                 #     All the knobs and dials
│   │   ├── types.py                  #     Shared types
│   │   └── pipeline.py               #     The main conductor
│   │
│   ├── analysis/                     #   Figuring out each song
│   │   ├── audio_loader.py           #     Reading files, measuring loudness
│   │   ├── spectral_analyzer.py      #     STFT, chroma, MFCC
│   │   ├── beat_analyzer.py          #     Tempo, downbeats, swing
│   │   ├── key_analyzer.py           #     Musical key detection
│   │   ├── vocal_analyzer.py         #     Where are the vocals?
│   │   ├── structure_analyzer.py     #     Intro, verse, chorus, outro
│   │   ├── genre_analyzer.py         #     What kind of music is this?
│   │   └── mood_analyzer.py          #     Happy? Sad? Energetic?
│   │
│   ├── mixing/                       #   The actual mixing
│   │   ├── transition_planner.py     #     Choosing style & parameters
│   │   ├── track_ordering.py         #     Smart playlist logic
│   │   ├── crossfader.py             #     6 different crossfade styles
│   │   ├── beat_aligner.py           #     Locking beats together
│   │   ├── tempo_sync.py             #     Phase vocoder BPM changes
│   │   ├── vocal_crossfade.py        #     LR4 frequency sweep magic
│   │   ├── transition_ding.py        #     Accent sound
│   │   └── assets/                   #     Audio assets
│   │
│   ├── acceleration/                 #   Making things fast
│   │   ├── metal_gpu.py              #     MPS GPU acceleration
│   │   └── parallel.py               #     Running analysis in parallel
│   │
│   └── cache/
│       └── analysis_cache.py         #   Caching so you don't re‑analyze
│
├── 💿 songs/otunedResult/            # Where the magic lands
│   ├── otune_mix.wav
│   ├── 001_*__*.wav
│   └── ...
│
├── 📜 src/                           # Legacy v1.x (rest in peace)
└── 🧪 tests/
    ├── test_refactored.py
    ├── test_transition_styles.py
    └── test_gpu_acceleration.py
```

---

## ⚙️ Configuration

All the knobs live in `otune_engine/core/config.py`:

| Config class | What you can tweak |
|---|---|
| `AnalysisConfig` | Sample rate, hop length, FFT size, target loudness, vocal frequency range |
| `TransitionConfig` | Crossfade duration (3–30 s), vocal mode, duck level, ding sound, tempo ramp % |
| `AccelerationConfig` | GPU on/off, batch size, worker threads |
| `CacheConfig` | Turn caching on/off, pick where to store it |

---

## 🧪 Running Tests

```bash
pytest tests/
```

GPU tests need PyTorch — `pip install -e ".[gpu]"`.

---

## 📄 License

**MIT** — do whatever you want with it, seriously.

Use it, modify it, sell it, put it in your own project, wrap it in a UI, turn it into a startup. The only thing we ask: keep the copyright notice in there somewhere.

**In plain English:** Free as in beer and speech — just don't remove our name.

---

<p align="center">
  <b>Credit:</b> TicTu Tech &nbsp;·&nbsp; <b>Developer:</b> Thanh Solar NEXT
</p>
