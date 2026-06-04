"""
Genre classification from audio features.

Scores tracks against genre profiles using tempo, harmonic/percussive
balance, spectral characteristics, and energy levels.
"""

import logging
from typing import Dict

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# Max possible weight per feature key (for normalized scoring)
FEATURE_WEIGHTS = {
    'tempo_range': 0.25, 'sweet_tempo': 0.10,
    'percussive_min': 0.25, 'harmonic_min': 0.30,
    'harmonic_range': 0.25, 'centroid_min': 0.15,
    'centroid_max': 0.15, 'centroid_range': 0.20,
    'energy_min': 0.15, 'contrast_min': 0.25,
    'contrast_var_min': 0.20, 'zcr_var_min': 0.15,
}

# Genre profile definitions: (feature, weight, range_or_condition)
GENRE_PROFILES = {
    'electronic': {'tempo_range': (115, 145), 'sweet_tempo': (125, 135),
                   'percussive_min': 0.52, 'centroid_min': 2500},
    'house':      {'tempo_range': (118, 132), 'sweet_tempo': (124, 128),
                   'percussive_min': 0.52, 'centroid_range': (2800, 4500)},
    'hiphop':     {'tempo_range': (80, 110), 'percussive_min': 0.50,
                   'centroid_max': 2500},
    'pop':        {'tempo_range': (95, 135), 'harmonic_range': (0.4, 0.6),
                   'centroid_range': (2000, 3500)},
    'rock':       {'tempo_range': (110, 170), 'harmonic_range': (0.35, 0.65),
                   'energy_min': 0.15},
    'jazz':       {'harmonic_min': 0.58, 'contrast_var_min': 4.5,
                   'zcr_var_min': 0.03},
    'classical':  {'harmonic_min': 0.70, 'contrast_min': 23},
    'vietnamese_ballad': {'tempo_range': (70, 95), 'harmonic_min': 0.60,
                          'centroid_max': 2800},
    'vietnamese_pop':    {'tempo_range': (95, 130), 'harmonic_range': (0.45, 0.55),
                          'centroid_range': (2200, 3200)},
    'cuba_bolero':       {'tempo_range': (60, 90), 'harmonic_min': 0.55,
                          'centroid_max': 3000},
    'future_funk':       {'tempo_range': (105, 128), 'sweet_tempo': (110, 120),
                          'harmonic_range': (0.4, 0.6), 'percussive_min': 0.4},
    'country':           {'tempo_range': (90, 145), 'sweet_tempo': (110, 130),
                          'harmonic_range': (0.48, 0.68)},
    'techno':    {'tempo_range': (130, 160), 'sweet_tempo': (138, 150),
                  'percussive_min': 0.55, 'centroid_min': 3000},
    'lofi':      {'tempo_range': (70, 95), 'harmonic_min': 0.60,
                  'centroid_max': 2200, 'energy_min': 0.12},
    'metal':     {'tempo_range': (140, 200), 'percussive_min': 0.58,
                  'energy_min': 0.20, 'centroid_min': 3000},
    'rb_soul':   {'tempo_range': (60, 95), 'harmonic_min': 0.55,
                  'centroid_range': (1800, 3000), 'contrast_var_min': 3.5},
    'reggae':    {'tempo_range': (65, 95), 'harmonic_range': (0.50, 0.70),
                  'centroid_max': 2500, 'percussive_min': 0.35},
}


def detect_genre(y_mono: np.ndarray, sr: int, tempo: float,
                 energy: float, spectral_centroid: float) -> str:
    """
    Detect genre hint from audio features.

    Args:
        y_mono: Mono audio signal.
        sr: Sample rate.
        tempo: Detected tempo (BPM).
        energy: Mean RMS energy.
        spectral_centroid: Mean spectral centroid (Hz).

    Returns:
        Genre string: 'electronic', 'pop', 'rock', 'jazz', 'classical',
        'hiphop', 'vietnamese_ballad', 'vietnamese_pop', 'cuba_bolero',
        'future_funk', 'house', 'country', 'techno', 'lofi', 'metal',
        'rb_soul', 'reggae', or 'unknown'.
    """
    # Limit to 90 seconds for speed
    max_samples = min(len(y_mono), sr * 90)
    y = y_mono[:max_samples]

    # Harmonic/percussive separation
    y_h, y_p = librosa.effects.hpss(y, margin=2.0)
    total_e = np.sqrt(np.mean(y ** 2)) + 1e-10
    h_ratio = np.sqrt(np.mean(y_h ** 2)) / total_e
    p_ratio = np.sqrt(np.mean(y_p ** 2)) / total_e
    total = h_ratio + p_ratio
    if total > 0:
        h_ratio /= total
        p_ratio /= total

    # Spectral contrast
    hop = 1024
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop)
    avg_contrast = float(np.mean(contrast))
    contrast_var = float(np.std(np.mean(contrast, axis=0)))

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]
    zcr_var = float(np.std(zcr))
    zcr_mean = float(np.mean(zcr))

    def _check(profile_key: str, weight_key: str, condition: bool) -> float:
        return FEATURE_WEIGHTS.get(weight_key, 0.0) if profile_key in profile and condition else 0.0

    # Score each genre
    scores: Dict[str, float] = {}

    for genre, profile in GENRE_PROFILES.items():
        score = 0.0

        # Tempo range
        if 'tempo_range' in profile:
            lo, hi = profile['tempo_range']
            score += _check('tempo_range', 'tempo_range', lo <= tempo <= hi)

        # Sweet tempo (independent of tempo_range)
        if 'sweet_tempo' in profile:
            slo, shi = profile['sweet_tempo']
            score += _check('sweet_tempo', 'sweet_tempo', slo <= tempo <= shi)

        # Percussive threshold
        score += _check('percussive_min', 'percussive_min', p_ratio > profile.get('percussive_min', 0))

        # Harmonic threshold
        score += _check('harmonic_min', 'harmonic_min', h_ratio > profile.get('harmonic_min', 0))

        # Harmonic range
        if 'harmonic_range' in profile:
            lo, hi = profile['harmonic_range']
            score += _check('harmonic_range', 'harmonic_range', lo <= h_ratio <= hi)

        # Spectral centroid
        score += _check('centroid_min', 'centroid_min', spectral_centroid > profile.get('centroid_min', 0))
        score += _check('centroid_max', 'centroid_max', spectral_centroid < profile.get('centroid_max', 0))
        if 'centroid_range' in profile:
            lo, hi = profile['centroid_range']
            score += _check('centroid_range', 'centroid_range', lo <= spectral_centroid <= hi)

        # Energy
        score += _check('energy_min', 'energy_min', energy > profile.get('energy_min', 0))

        # Contrast
        score += _check('contrast_min', 'contrast_min', avg_contrast > profile.get('contrast_min', 0))
        score += _check('contrast_var_min', 'contrast_var_min', contrast_var > profile.get('contrast_var_min', 0))

        # ZCR
        score += _check('zcr_var_min', 'zcr_var_min', zcr_var > profile.get('zcr_var_min', 0))

        max_possible = sum(FEATURE_WEIGHTS.get(k, 0.0) for k in profile)
        if max_possible > 0:
            score_norm = score / max_possible
        else:
            score_norm = 0.0

        if score_norm >= 0.55:
            scores[genre] = score_norm

    if not scores:
        return 'unknown'

    # Return highest scoring genre (normalized to 0-1)
    sorted_genres = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_genre, top_score = sorted_genres[0]

    if top_score >= 0.55:
        ambiguous = ''
        if len(sorted_genres) > 1 and top_score - sorted_genres[1][1] < 0.15:
            ambiguous = f', ambiguous with {sorted_genres[1][0]}'
        logger.info(f"  Genre: {top_genre} (conf: {top_score:.2f}{ambiguous})")
        return top_genre

    return 'unknown'
