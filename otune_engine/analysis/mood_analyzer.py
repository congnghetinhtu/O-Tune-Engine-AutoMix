"""
Mood estimation from musical features.

Returns a mood score in [0, 1] where 0 is sad and 1 is happy.
"""

from typing import Tuple


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _norm(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return _clamp((v - lo) / (hi - lo), 0.0, 1.0)


def _key_signature_bias(key_index: int, key_mode: str) -> float:
    """Return brightness bias from key signature (-1..1)."""
    # Map pitch class to signed accidentals (sharps positive, flats negative)
    # Prefer the representation with fewer accidentals.
    accidental_map = {
        0: 0,    # C
        7: 1,    # G
        2: 2,    # D
        9: 3,    # A
        4: 4,    # E
        11: 5,   # B
        6: 6,    # F#/Gb
        5: -1,   # F
        10: -2,  # Bb
        3: -3,   # Eb
        8: -4,   # Ab
        1: -5,   # Db
    }

    pc = int(key_index) % 12
    if str(key_mode).lower().startswith('min'):
        # Use relative major for key signature
        pc = (pc + 3) % 12

    acc = accidental_map.get(pc, 0)
    return _clamp(acc / 7.0, -1.0, 1.0)


def estimate_mood(tempo: float, energy: float, energy_variation: float,
                  spectral_centroid: float, spectral_rolloff: float,
                  zcr: float, key_index: int, key_mode: str,
                  key_confidence: float = 0.0) -> Tuple[float, str]:
    """
    Estimate mood using arousal (tempo/energy) and valence (brightness/mode).

    Args:
        tempo: BPM.
        energy: Mean RMS energy (0..~0.3 typical).
        energy_variation: RMS variance over time.
        spectral_centroid: Average spectral centroid in Hz.
        spectral_rolloff: Average spectral rolloff in Hz.
        zcr: Zero-crossing rate.
        key_mode: 'major' or 'minor'.
        key_confidence: 0..1 confidence of the detected key.

    Returns:
        (mood_score, mood_label)
    """
    tempo_n = _norm(tempo, 60.0, 180.0)
    energy_n = _norm(energy, 0.02, 0.25)
    energy_var_n = _norm(energy_variation, 0.0, 0.08)
    bright_n = _norm(spectral_centroid, 1200.0, 4500.0)
    roll_n = _norm(spectral_rolloff, 1800.0, 8000.0)
    zcr_n = _norm(zcr, 0.02, 0.15)

    mode_weight = 0.5 + 0.5 * _clamp(key_confidence, 0.0, 1.0)
    mode_bonus = (0.12 if str(key_mode).lower().startswith('maj') else -0.12) * mode_weight
    tone_bias = _key_signature_bias(key_index, key_mode) * (0.06 * mode_weight)

    # Valence favors brighter timbre, stable energy, and major keys
    stability = 1.0 - energy_var_n
    valence = (bright_n * 0.35) + (roll_n * 0.20) + (stability * 0.20) + ((1.0 - zcr_n) * 0.10) + mode_bonus + tone_bias
    valence = _clamp(valence, 0.0, 1.0)

    # Arousal favors tempo and energy
    arousal = (tempo_n * 0.55) + (energy_n * 0.45)
    arousal = _clamp(arousal, 0.0, 1.0)

    mood = (valence * 0.60) + (arousal * 0.40)
    mood = _clamp(mood, 0.0, 1.0)

    if mood < 0.30:
        label = 'sad'
    elif mood < 0.48:
        label = 'calm'
    elif mood < 0.68:
        label = 'neutral'
    else:
        label = 'happy'

    return mood, label
