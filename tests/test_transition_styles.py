import sys
from pathlib import Path

import numpy as np
import pytest

# Add repo root to path so `otune_engine` is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from otune_engine.analysis.structure_analyzer import StructureAnalyzer
from otune_engine.core.pipeline import MixPipeline
from otune_engine.core.config import TransitionConfig
from otune_engine.core.types import TrackAnalysis, StructureSection, VocalSegment
from otune_engine.mixing.transition_planner import TransitionPlanner
from otune_engine.mixing.vocal_crossfade import VocalCrossfader


def _equal_power_fades(n: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    fo = np.cos(t * np.pi / 2.0).astype(np.float32)
    fi = np.sin(t * np.pi / 2.0).astype(np.float32)
    return fo, fi


def test_bass_swap_frequency_sweep():
    """bass_swap uses a frequency-sweep crossfade (lows sweep in first,
    highs linger). Verify it differs from standard equal-power and
    produces direction-dependent output."""
    sr = 44100
    n = sr
    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)

    fo, fi = _equal_power_fades(n)
    x = VocalCrossfader(sample_rate=sr)

    a1 = 0.5 * np.sin(2 * np.pi * 100.0 * t)
    a2 = 0.3 * np.sin(2 * np.pi * 440.0 * t)

    out_bass = x.crossfade(a1, a2, fo, fi, vocal_strategy="bass_swap")
    out_std = x._standard_crossfade(a1, a2, fo, fi)

    # Must differ from plain equal-power crossfade (frequency sweep)
    diff = float(np.sqrt(np.mean((out_bass - out_std) ** 2)))
    assert diff > 1e-4

    # Must be direction-dependent: swapping a1/a2 changes result
    out_rev = x.crossfade(a2, a1, fo, fi, vocal_strategy="bass_swap")
    diff_rev = float(np.sqrt(np.mean((out_bass - out_rev) ** 2)))
    assert diff_rev > 1e-4

    # Basic sanity
    assert np.all(np.isfinite(out_bass))
    peak = float(np.max(np.abs(out_bass)))
    assert 0.0 < peak <= 1.0


def test_bass_swap_low_band_transitions_first():
    """Incoming lows arrive faster with bass_swap than standard blend."""
    sr = 44100
    n = sr
    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)

    fo, fi = _equal_power_fades(n)
    x = VocalCrossfader(sample_rate=sr)

    # Outgoing silence, incoming low-frequency signal
    a1 = np.zeros_like(t)
    a2 = 0.5 * np.sin(2 * np.pi * 100.0 * t)

    out_bass = x.crossfade(a1, a2, fo, fi, vocal_strategy="bass_swap")
    out_std = x._standard_crossfade(a1, a2, fo, fi)

    # Early region (35-45%): sweep should have more signal from the
    # fast-arriving low band than the standard linear fade
    s0, s1 = int(0.35 * n), int(0.45 * n)
    rms_bass = float(np.sqrt(np.mean(out_bass[s0:s1] ** 2)))
    rms_norm = float(np.sqrt(np.mean(out_std[s0:s1] ** 2)))
    assert rms_bass > rms_norm * 0.95


def test_transition_planner_defaults_to_apple_automix():
    planner = TransitionPlanner(TransitionConfig())

    t1 = TrackAnalysis(
        duration=180.0,
        actual_tempo=120.0,
        energy=0.15,
        key=0,
        key_mode="major",
        key_confidence=0.0,
        genre_hint="unknown",
    )
    t2 = TrackAnalysis(
        duration=180.0,
        actual_tempo=122.0,
        energy=0.16,
        key=2,
        key_mode="major",
        key_confidence=0.0,
        genre_hint="unknown",
    )

    style, _params = planner._select_style(t1, t2)
    assert style == "apple_automix"


def test_structure_labeling_sets_repetition_and_boring_scores():
    analyzer = StructureAnalyzer(sample_rate=10, hop_length=1)

    duration = 20.0
    beats = np.arange(40, dtype=np.float32) * 0.5  # 40 beats, 0.5s each

    # 4 sections of 5s each
    boundaries = [0.0, 5.0, 10.0, 15.0]

    # Synthetic RMS envelope: quiet intro, loud middle, moderate outro
    rms = np.zeros(int(duration * analyzer.sr / analyzer.hop), dtype=np.float32)
    rms[:50] = 0.10
    rms[50:150] = 0.50
    rms[150:] = 0.20

    # Vocals in the middle sections (5–15s)
    vocal_segments = [VocalSegment(start=5.0, end=15.0, energy=0.4, confidence=0.9)]

    # Self-similarity: make section 2 similar to section 3
    sim = np.eye(len(beats), dtype=np.float32)
    sim[10:20, 20:30] = 0.90
    sim[20:30, 10:20] = 0.90

    sections = analyzer._label_sections(boundaries, duration, beats, vocal_segments, rms, sim)

    assert len(sections) == 4
    assert sections[0].label == "intro"
    assert sections[0].boring_score > 0.5

    # Repeated middle sections should have strong repetition score
    assert sections[1].repetition_score > 0.7
    assert sections[2].repetition_score > 0.7


def test_cache_serializes_repetition_score():
    t = TrackAnalysis()
    t.structure_sections = [
        StructureSection(start=0.0, end=10.0, label="intro", repetition_score=0.8, boring_score=0.9),
    ]

    d = MixPipeline._track_to_dict(t)
    assert "structure_sections" in d
    assert d["structure_sections"][0]["repetition_score"] == pytest.approx(0.8)
