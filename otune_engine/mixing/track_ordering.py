"""
Smart track ordering with look-ahead optimization.

Determines the best playback order using greedy selection
with forward compatibility checking to avoid dead ends.
"""

import logging
from typing import List, Optional

from ..core.types import TrackAnalysis
from .transition_planner import TransitionPlanner

logger = logging.getLogger(__name__)


class TrackOrderer:
    """Determine optimal track playback order."""

    def __init__(self, planner: TransitionPlanner):
        self.planner = planner

    def order(self, tracks: List[TrackAnalysis],
              start_index: Optional[int] = None) -> List[TrackAnalysis]:
        """
        Reorder tracks for optimal transitions.

        Uses greedy selection with 1-step look-ahead to avoid
        getting stuck with incompatible final tracks.
        """
        if len(tracks) <= 1:
            return tracks

        # Mood range for sad-to-happy progression
        mood_scores = [getattr(t, 'mood_score', 0.5) for t in tracks]
        mood_min = min(mood_scores)
        mood_max = max(mood_scores)
        mood_span = max(0.0, mood_max - mood_min)
        use_mood = mood_span >= 0.15

        # Select starting track
        if start_index is not None and 0 <= start_index < len(tracks):
            start = tracks[start_index]
            remaining = [t for i, t in enumerate(tracks) if i != start_index]
            logger.info(f"Starting with: #{start_index + 1} {start.file_path.name}")
        else:
            start, remaining = self._auto_start(tracks)
            logger.info(f"Auto-selected start: {start.file_path.name}")

        ordered = [start]

        while remaining:
            current = ordered[-1]
            best_track = None
            best_score = -1.0

            progress = len(ordered) / max(len(tracks) - 1, 1)
            target_mood = mood_min + progress * mood_span if use_mood else None
            current_mood = getattr(current, 'mood_score', 0.5)

            for candidate in remaining:
                compat = self.planner.compatibility(current, candidate)

                # Look-ahead: check if this leaves reasonable options
                if len(remaining) > 1:
                    others = [t for t in remaining if t != candidate]
                    fwd = sum(
                        self.planner.compatibility(candidate, o) for o in others
                    ) / len(others)
                    mood_score = 0.0
                    if use_mood and target_mood is not None:
                        mood_score = 1.0 - abs(candidate.mood_score - target_mood) / max(mood_span, 1e-6)
                        mood_score = max(0.0, min(1.0, mood_score))
                    score = compat * 0.58 + fwd * 0.22 + mood_score * 0.20
                else:
                    mood_score = 0.0
                    if use_mood and target_mood is not None:
                        mood_score = 1.0 - abs(candidate.mood_score - target_mood) / max(mood_span, 1e-6)
                        mood_score = max(0.0, min(1.0, mood_score))
                    score = compat * 0.70 + mood_score * 0.30

                # Discourage stepping backwards in mood progression
                if use_mood and candidate.mood_score < (current_mood - 0.03):
                    score -= 0.12

                if score > best_score:
                    best_score = score
                    best_track = candidate

            if best_track:
                ordered.append(best_track)
                remaining.remove(best_track)
                if use_mood:
                    logger.info(f"  → {best_track.file_path.name} "
                               f"(score: {best_score:.3f}, mood: {best_track.mood_score:.2f})")
                else:
                    logger.info(f"  → {best_track.file_path.name} "
                               f"(score: {best_score:.3f})")
            else:
                ordered.append(remaining.pop(0))

        return ordered

    def _auto_start(self, tracks: List[TrackAnalysis]):
        """Select best starting track automatically."""
        mood_scores = [getattr(t, 'mood_score', 0.5) for t in tracks]
        mood_min = min(mood_scores)
        mood_max = max(mood_scores)
        mood_span = max(0.0, mood_max - mood_min)
        use_mood = mood_span >= 0.15

        best = None
        best_score = -1.0
        best_idx = 0

        for i, t in enumerate(tracks):
            score = 0.0

            # Moderate energy (25%)
            if 0.10 <= t.energy <= 0.18:
                score += 0.25
            elif 0.08 <= t.energy <= 0.22:
                score += 0.12

            # Moderate tempo (20%)
            if 95 <= t.actual_tempo <= 130:
                score += 0.20
            elif 80 <= t.actual_tempo <= 140:
                score += 0.08

            # Key compatibility with most tracks (20%)
            compat_count = sum(
                1 for o in tracks if o != t and
                min(abs(t.key - o.key), 12 - abs(t.key - o.key)) in (0, 3, 5, 7)
            )
            score += (compat_count / max(len(tracks) - 1, 1)) * 0.20

            # Versatile genre (15%)
            if t.genre_hint in ('pop', 'electronic', 'house'):
                score += 0.15
            elif t.genre_hint in ('hiphop', 'future_funk'):
                score += 0.10

            # Avg compatibility (20%)
            avg_c = sum(
                self.planner.compatibility(t, o) for o in tracks if o != t
            ) / max(len(tracks) - 1, 1)
            score += avg_c * 0.20

            # Prefer lower mood for the start (sad → happy)
            if use_mood:
                mood_norm = (t.mood_score - mood_min) / max(mood_span, 1e-6)
                score += (1.0 - mood_norm) * 0.15

            if score > best_score:
                best_score = score
                best = t
                best_idx = i

        remaining = [t for i, t in enumerate(tracks) if i != best_idx]
        return best, remaining
