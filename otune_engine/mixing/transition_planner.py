"""
Transition planning — choose crossfade style and parameters.

Selects from: smooth_blend, energy_punch, harmonic_layer,
build_drop, palate_cleanser, apple_automix based on musical context.
"""

import logging
from typing import Optional, Tuple
import numpy as np

from ..core.types import TrackAnalysis, TransitionPlan
from ..core.config import TransitionConfig

logger = logging.getLogger(__name__)


class TransitionPlanner:
    """Choose transition style and parameters for each track pair."""

    def __init__(self, config: TransitionConfig):
        self.config = config

    def plan(self, t1: TrackAnalysis, t2: TrackAnalysis) -> TransitionPlan:
        """Create a full transition plan between two tracks."""
        compat = self.compatibility(t1, t2)
        style, params = self._select_style(t1, t2)
        duration = self._optimal_duration(t1, t2, style)
        duration = float(np.clip(duration, self.config.min_crossfade,
                                  self.config.max_crossfade))
        xfade_start, intro_skip = self._transition_points(t1, t2, duration)

        plan = TransitionPlan(
            style=style,
            crossfade_duration=duration,
            crossfade_start=xfade_start,
            intro_skip=intro_skip,
            fade_curve_power=params.get('curve', 1.0),
            gap_duration=params.get('gap', 0.0),
            overlap_boost=params.get('overlap', 0.5),
            confidence=compat,
            reason=params.get('reason', ''),
            compatibility_score=compat,
        )
        logger.info(f"  Plan: '{style}' | {duration:.1f}s | compat={compat:.2f}")
        return plan

    def compatibility(self, t1: TrackAnalysis, t2: TrackAnalysis) -> float:
        """Calculate 0-1 compatibility score between tracks."""
        # Tempo (30%)
        t1_tempo = float(getattr(t1, 'actual_tempo', 0.0) or 0.0)
        t2_tempo = float(getattr(t2, 'actual_tempo', 0.0) or 0.0)
        if t1_tempo <= 0.0 or t2_tempo <= 0.0:
            ts = 0.5
        else:
            td = abs(t1_tempo - t2_tempo)
            ratio = max(t1_tempo, t2_tempo) / min(t1_tempo, t2_tempo)
            if abs(ratio - 2.0) < 0.1 or abs(ratio - 1.5) < 0.1:
                ts = 0.9
            else:
                tol = 15 if t1.genre_hint == t2.genre_hint else 20
                ts = max(0, 1 - td / tol)

        # Key (25%)
        ks = self._key_compatibility(t1, t2)
        conf = min(float(t1.key_confidence), float(t2.key_confidence))
        conf_w = 0.5 + 0.5 * np.clip(conf, 0.0, 1.0)
        ks = 0.5 + (ks - 0.5) * conf_w

        # Energy (20%)
        ed = abs(t1.energy - t2.energy)
        es = max(0, 1 - ed / max(t1.energy, t2.energy, 0.1))

        # Spectral (10%)
        scs = max(0, 1 - abs(t1.spectral_centroid - t2.spectral_centroid) / 2000)

        # Timbral (10%)
        md = np.linalg.norm(t1.mfcc_mean - t2.mfcc_mean)
        ms = max(0, 1 - md / 50)

        # Groove (5%)
        gb = 0.05 if t1.groove_type == t2.groove_type != 'straight' else 0

        return min(1.0, ts * 0.30 + ks * 0.25 + es * 0.20 + scs * 0.10 + ms * 0.10 + gb)

    def _select_style(self, t1: TrackAnalysis, t2: TrackAnalysis) -> Tuple[str, dict]:
        """Select transition style based on musical context."""
        ks = self._key_compatibility(t1, t2)
        ac = (t1.key_confidence + t2.key_confidence) / 2

        if ks <= 0.25 and ac > 0.7:
            return 'palate_cleanser', {'gap': 1.5, 'curve': 0.9, 'reason': 'key_clash'}

        if ks >= 0.85 and ac > 0.6:
            return 'harmonic_layer', {'curve': 0.5, 'overlap': 0.75,
                                      'reason': 'harmonic_match'}

        ej = t2.energy - t1.energy
        if t1.energy < 0.12 and t2.energy > 0.18 and ej > 0.08:
            return 'energy_punch', {'gap': 0.3, 'curve': 1.2,
                                    'reason': f'energy_jump_{ej:.2f}'}

        if t1.energy >= 0.12 and t2.energy > 0.20 and ej > 0.06:
            return 'build_drop', {'gap': 0.2, 'curve': 1.1,
                                  'reason': f'build_drop_{ej:.2f}'}

        # Default: Apple-style phrase blend with bass swap — uses same
        # gentle fade curve as harmonic_layer for a smooth, seamless blend.
        return 'apple_automix', {'curve': 0.5, 'overlap': 0.5,
                                 'reason': f'otune_{t1.genre_hint}_to_{t2.genre_hint}'}

    def _optimal_duration(self, t1: TrackAnalysis, t2: TrackAnalysis, style: str) -> float:
        """Calculate optimal crossfade duration — phrase-aware and genre-adaptive.

        For 'apple_automix' and 'harmonic_layer', snaps to the nearest
        musical phrase boundary (8 or 16 bars) for seamless phrasing.
        """
        base = self.config.crossfade_duration
        td = abs(t1.actual_tempo - t2.actual_tempo)

        # For extreme tempo ratios (>1.4), force minimum crossfade
        # since even the tempo ramp can't fully close the gap.
        t1_t = float(getattr(t1, 'actual_tempo', 0.0) or 0.0)
        t2_t = float(getattr(t2, 'actual_tempo', 0.0) or 0.0)
        if t1_t > 0 and t2_t > 0:
            tempo_ratio = max(t1_t, t2_t) / min(t1_t, t2_t)
        else:
            tempo_ratio = 1.0
        if tempo_ratio >= 1.4:
            return self.config.min_crossfade

        if td < 5:
            dur = base * 1.5
        elif td < 10:
            dur = base * 1.3
        elif td < 20:
            dur = base * 1.0
        else:
            dur = base * 0.9

        genre_scale = {'jazz': 1.3, 'classical': 1.3, 'hiphop': 0.8,
                       'vietnamese_ballad': 1.2, 'future_funk': 0.9,
                       'pop': 1.05, 'electronic': 1.1}
        dur *= genre_scale.get(t1.genre_hint, 1.0)

        # Re-clamp after genre scaling so max_crossfade is truly respected
        dur = min(dur, self.config.max_crossfade)

        if style == 'energy_punch':
            dur = min(dur, 5.0)
        elif style in ('apple_automix', 'harmonic_layer'):
            dur = max(dur, 6.0)

        # Snap to nearest full bar so crossfade starts and ends on a beat
        tempo = max(t1.actual_tempo, t2.actual_tempo, 1.0)
        meter = max(t1.time_signature, t2.time_signature, 4)
        bar_dur = (60.0 / tempo) * meter
        if bar_dur > 0:
            n_bars = max(1, round(dur / bar_dur))
            dur = n_bars * bar_dur
        return dur

    def _transition_points(self, t1: TrackAnalysis, t2: TrackAnalysis,
                           duration: float) -> Tuple[float, float]:
        """Find crossfade start and intro skip using structure, phrases, and beats.

        Prioritises musical boundaries in this order:
          1. Phrase boundaries (clean musical handoff)
          2. Downbeats (measure boundaries)
          3. Outro sections
          4. Strong beats near vocal gaps
        """
        max_start = max(0.0, t1.duration - duration)
        xfade = max_start

        phrase = self._pick_phrase_boundary(t1, max_start)
        if phrase is not None:
            xfade = phrase
            # Phrase boundary is the strongest musical alignment — only
            # apply a final beat snap within a tight window so we don't
            # undo the clean phrase handoff.
            xfade = self._snap_to_beat(t1, xfade)
        else:
            xfade = self._pick_structure_boundary(t1, max_start, xfade)

            xfade = self._snap_to_downbeat(t1, xfade, max_start)

            xfade = self._snap_to_vocal_gap(t1, xfade)

            xfade = self._snap_to_beat(t1, xfade)

        xfade = float(np.clip(xfade, 0.0, max_start))

        intro_skip = self._compute_intro_skip(t1, t2, duration, xfade)

        return xfade, intro_skip

    def _pick_phrase_boundary(self, track: TrackAnalysis, max_start: float) -> Optional[float]:
        """Stage 1: pick the latest phrase boundary before max_start."""
        if not getattr(track, 'phrases', None):
            return None
        phrase_ends = [
            float(p[1]) for p in (track.phrases or [])
            if 0.0 < float(p[1]) <= max_start
        ]
        if not phrase_ends:
            return None
        return float(max(phrase_ends))

    def _pick_structure_boundary(self, track: TrackAnalysis, max_start: float,
                                  fallback: float) -> float:
        """Stage 2: use outro or section boundary when no phrase is available."""
        xfade = fallback
        if not (track.structure_sections or []):
            outro = float(getattr(track, 'outro_start', 0.0) or 0.0)
            if 0.0 < outro <= max_start:
                xfade = float(np.clip(max(xfade, outro), 0.0, max_start))

        for sec in track.structure_sections:
            if sec.label == 'outro':
                xfade = sec.start + sec.duration * 0.6
                break

        boundary = self._find_section_boundary(track, max_start)
        if boundary is not None and abs(boundary - xfade) <= 8.0:
            xfade = boundary
        return xfade

    def _snap_to_downbeat(self, track: TrackAnalysis, xfade: float,
                           max_start: float) -> float:
        """Stage 3: snap crossfade start to the nearest downbeat."""
        if len(track.downbeats) == 0:
            return xfade
        db_candidates = track.downbeats[(track.downbeats >= xfade - 2.0) &
                                         (track.downbeats <= max_start)]
        if len(db_candidates) > 0:
            return float(db_candidates[0])

        idx = int(np.argmin(np.abs(track.downbeats - xfade)))
        db = float(track.downbeats[idx])
        tempo = float(getattr(track, 'actual_tempo', 0.0) or 0.0)
        meter = int(getattr(track, 'time_signature', 4) or 4)
        if tempo > 0.0:
            beat_dur = 60.0 / tempo
            bar_dur = beat_dur * max(1, meter)
            max_snap = float(np.clip(bar_dur * 2.0, 2.0, 8.0))
        else:
            max_snap = 4.0
        if abs(db - xfade) <= max_snap and db <= max_start:
            return db
        return xfade

    def _snap_to_vocal_gap(self, track: TrackAnalysis, xfade: float) -> float:
        """Stage 4: shift toward a nearby vocal gap to reduce vocal clash."""
        if not track.has_vocals or not track.vocal_segments:
            return xfade
        gap = self._find_gap_near(track.vocal_segments, xfade, window=2.0)
        if gap is None:
            return xfade
        g_start, g_end = gap
        if g_start >= xfade and (g_start - xfade) <= 1.5:
            return g_start
        if g_end <= xfade and (xfade - g_end) <= 0.5:
            return g_start
        return xfade

    def _snap_to_beat(self, track: TrackAnalysis, xfade: float) -> float:
        """Stage 5: final snap to the nearest beat if close enough."""
        if len(track.beats) == 0:
            return xfade
        idx = int(np.argmin(np.abs(track.beats - xfade)))
        beat_delta = float(abs(track.beats[idx] - xfade))
        tempo = float(getattr(track, 'actual_tempo', 0.0) or 0.0)
        if tempo > 0.0:
            beat_dur = 60.0 / tempo
            max_snap = float(np.clip(beat_dur * 2.0, 0.6, 2.5))
        else:
            max_snap = 1.5
        if beat_delta <= max_snap:
            return float(track.beats[idx])
        return xfade

    def _compute_intro_skip(self, t1: TrackAnalysis, t2: TrackAnalysis,
                            duration: float, xfade: float) -> float:
        """Compute intro skip for track 2 — structure-aware with beat snapping."""
        intro_skip = 0.0
        for sec in t2.structure_sections:
            if sec.label == 'intro' and sec.boring_score > 0.75:
                for s2 in t2.structure_sections:
                    if s2.label == 'verse':
                        intro_skip = max(0, s2.start - 0.5)
                        break
                if intro_skip == 0:
                    intro_skip = min(sec.end * 0.4, 4.0)
                break

        if intro_skip <= 0.0 and not (t2.structure_sections or []):
            intro_end = float(getattr(t2, 'intro_end', 0.0) or 0.0)
            if intro_end >= 8.0 and t2.has_vocals and t2.vocal_segments:
                first_vocal = float(min(v.start for v in t2.vocal_segments))
                if first_vocal > intro_end + 1.0:
                    intro_skip = max(0.0, min(intro_end, first_vocal) - 0.5)

        if t2.has_vocals and t2.vocal_segments:
            max_skip = float(np.clip(
                min(6.0, max(0.0, t2.duration - duration)), 0.0, t2.duration
            ))
            gap = self._find_early_gap(t2.vocal_segments, intro_skip, max_skip)
            if gap is not None:
                g_start, _ = gap
                if g_start > intro_skip + 0.4 and g_start <= intro_skip + 2.0:
                    intro_skip = g_start

        intro_skip = min(intro_skip, 6.0)
        buffer = max(0.5, duration * 0.1)  # leave room for post-crossfade audio
        intro_skip = float(np.clip(intro_skip, 0.0, max(0.0, t2.duration - duration - buffer)))

        if len(t2.downbeats) > 0:
            db2 = t2.downbeats[(t2.downbeats >= intro_skip) &
                                (t2.downbeats <= intro_skip + 2.0)]
            if len(db2) > 0:
                intro_skip = float(db2[0])
            else:
                idx = int(np.argmin(np.abs(t2.downbeats - intro_skip)))
                if abs(float(t2.downbeats[idx]) - intro_skip) <= 1.5:
                    intro_skip = float(t2.downbeats[idx])
        elif len(t2.beats) > 0:
            idx = int(np.argmin(np.abs(t2.beats - intro_skip)))
            if abs(float(t2.beats[idx]) - intro_skip) <= 1.0:
                intro_skip = float(t2.beats[idx])

        return intro_skip

    @staticmethod
    def _key_distance(k1: int, k2: int) -> int:
        d = abs(int(k1) - int(k2)) % 12
        return min(d, 12 - d)

    def _key_compatibility(self, t1: TrackAnalysis, t2: TrackAnalysis) -> float:
        kd = self._key_distance(t1.key, t2.key)
        same_mode = (t1.key_mode == t2.key_mode)

        # Relative major/minor
        rel = False
        if t1.key_mode != t2.key_mode:
            if t1.key_mode == 'minor' and (t1.key + 3) % 12 == t2.key:
                rel = True
            elif t2.key_mode == 'minor' and (t2.key + 3) % 12 == t1.key:
                rel = True

        if kd == 0 and same_mode:
            return 1.0
        if rel:
            return 0.88
        if kd == 0 and not same_mode:
            return 0.78
        if kd in (5, 7) and same_mode:
            return 0.8
        if kd in (2, 4, 8, 10):
            return 0.6
        if kd == 6:
            return 0.2
        return max(0.0, 1.0 - (kd / 7.0))

    @staticmethod
    def _find_section_boundary(track: TrackAnalysis, max_start: float) -> Optional[float]:
        sections = getattr(track, 'structure_sections', []) or []
        if not sections:
            return None

        candidates = []
        for sec in sections:
            end_t = float(sec.end)
            if end_t <= 0 or end_t > max_start:
                continue
            if sec.label not in ('outro', 'chorus', 'verse', 'bridge'):
                continue
            # Prefer later boundaries and lower-energy endings
            label_bonus = 0.2 if sec.label == 'outro' else 0.1 if sec.label == 'chorus' else 0.0
            late_bonus = end_t / max(max_start, 1e-6)
            energy_penalty = float(np.clip(sec.energy, 0.0, 1.0))
            score = (late_bonus * 0.6) + label_bonus - (energy_penalty * 0.2)
            candidates.append((score, end_t))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return float(candidates[0][1])

    @staticmethod
    def _vocal_gaps(segments, min_gap: float = 0.3, max_gap: float = 6.0):
        gaps = []
        if not segments or len(segments) < 2:
            return gaps
        for i in range(len(segments) - 1):
            start = float(segments[i].end)
            end = float(segments[i + 1].start)
            dur = end - start
            if min_gap <= dur <= max_gap:
                gaps.append((start, end))
        return gaps

    def _find_gap_near(self, segments, center: float, window: float = 3.0):
        gaps = self._vocal_gaps(segments)
        if not gaps:
            return None
        candidates = [g for g in gaps if (center - window) <= g[0] <= (center + window)]
        if not candidates:
            return None
        return min(candidates, key=lambda g: abs(g[0] - center))

    def _find_early_gap(self, segments, min_time: float, max_time: float):
        gaps = self._vocal_gaps(segments)
        if not gaps:
            return None
        candidates = [g for g in gaps if min_time <= g[0] <= max_time]
        if not candidates:
            return None
        return min(candidates, key=lambda g: g[0])


