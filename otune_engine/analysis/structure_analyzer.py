"""
Song structure detection (intro/verse/chorus/bridge/outro).

Uses self-similarity matrix, novelty detection, and energy/vocal
analysis to label sections and identify the main (chorus) section.
"""

import logging
from typing import List, Optional

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks

from ..core.types import VocalSegment, StructureSection

logger = logging.getLogger(__name__)


class StructureAnalyzer:
    """Detect and label song structure sections."""

    def __init__(self, sample_rate: int = 44100, hop_length: int = 512):
        self.sr = sample_rate
        self.hop = hop_length

    def analyze(self, y_mono: np.ndarray, beats: np.ndarray,
                chroma: np.ndarray, mfccs: np.ndarray,
                rms: np.ndarray,
                vocal_segments: List[VocalSegment]) -> dict:
        """
        Full structure analysis.

        Returns dict with:
            structure_sections: List[StructureSection]
            main_section: Optional[StructureSection]
        """
        duration = len(y_mono) / self.sr

        # Skip for very short tracks
        if duration < 60.0 or len(beats) < 16:
            logger.info(f"  Skipping structure analysis (too short)")
            return {
                'structure_sections': [],
                'main_section': None,
            }

        # Step 1: Self-similarity matrix
        sim_matrix = self._self_similarity(chroma, mfccs, beats)

        # Step 2: Detect structural boundaries
        boundaries = self._detect_boundaries(
            y_mono, beats, sim_matrix
        )

        # Step 3: Label sections
        sections = self._label_sections(
            boundaries, duration, beats, vocal_segments, rms, sim_matrix
        )

        # Step 4: Identify main section (strongest chorus)
        main = self._identify_main(sections, beats, vocal_segments, rms, sim_matrix)

        logger.info(f"  Structure: {' → '.join(s.label for s in sections)}")
        return {
            'structure_sections': sections,
            'main_section': main,
        }

    # ── Self-Similarity Matrix ──────────────────────────────────────

    def _self_similarity(self, chroma: np.ndarray, mfccs: np.ndarray,
                         beats: np.ndarray) -> np.ndarray:
        """Beat-synchronized self-similarity matrix."""
        try:
            # librosa's sync utilities expect frame indices; prefer an explicit
            # time→frame conversion with the analyzer's hop length.
            beat_frames = librosa.time_to_frames(beats, sr=self.sr, hop_length=self.hop)
            beat_frames = np.clip(beat_frames, 0, max(0, chroma.shape[1] - 1))
            beat_frames = np.unique(beat_frames)
            if len(beat_frames) < 4:
                return np.eye(len(beats))

            # Include trailing region so every beat interval maps to a sync column.
            # librosa.util.sync produces len(beat_frames)-1 columns by default,
            # discarding the final beat→end region.
            end_frame = max(0, chroma.shape[1] - 1)
            if beat_frames[-1] < end_frame:
                beat_frames = np.append(beat_frames, end_frame)

            chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
            mfcc_sync = librosa.util.sync(mfccs, beat_frames, aggregate=np.median)

            # Normalize
            def norm_feat(x):
                mn = np.min(x, axis=1, keepdims=True)
                mx = np.max(x, axis=1, keepdims=True)
                return (x - mn) / (mx - mn + 1e-8)

            combined = np.vstack([
                norm_feat(chroma_sync) * 0.6,
                norm_feat(mfcc_sync[:5, :]) * 0.4
            ])

            sim = np.dot(combined.T, combined)
            sim = gaussian_filter(sim, sigma=1.5)
            np.fill_diagonal(sim, 1.0)
            sim = np.clip(sim, 0, 1)

            # Robustness: ensure the matrix aligns with `beats` indexing.
            target = int(len(beats))
            if target <= 0:
                return sim
            if sim.shape[0] == target:
                return sim

            n = int(min(sim.shape[0], target))
            fixed = np.eye(target, dtype=sim.dtype)
            fixed[:n, :n] = sim[:n, :n]
            return fixed

        except Exception as e:
            logger.warning(f"  Self-similarity failed: {e}")
            n = len(beats) if len(beats) > 0 else 100
            return np.eye(n)

    # ── Boundary Detection ──────────────────────────────────────────

    def _detect_boundaries(self, y_mono: np.ndarray,
                           beats: np.ndarray,
                           sim_matrix: np.ndarray) -> List[float]:
        """Detect major structural change points via novelty."""
        try:
            if sim_matrix.shape[0] < 16:
                return []

            lag = librosa.segment.recurrence_to_lag(sim_matrix, pad=False, axis=0)
            novelty = np.sum(lag, axis=0)
            novelty = gaussian_filter1d(novelty, sigma=3)

            threshold = np.mean(novelty) + 0.5 * np.std(novelty)
            peaks, _ = find_peaks(novelty, height=threshold, distance=8)

            boundaries = []
            for idx in peaks:
                if idx < len(beats):
                    boundaries.append(float(beats[idx]))

            # Filter too-close boundaries
            filtered = []
            last = -10.0
            for b in sorted(boundaries):
                if b - last >= 8.0:
                    filtered.append(b)
                    last = b

            if not filtered or filtered[0] > 1.0:
                filtered.insert(0, 0.0)

            return filtered

        except Exception as e:
            logger.warning(f"  Boundary detection failed: {e}")
            duration = len(y_mono) / self.sr
            return [0.0, duration * 0.25, duration * 0.5, duration * 0.75]

    # ── Section Labeling ────────────────────────────────────────────

    def _label_sections(self, boundaries: List[float], duration: float,
                        beats: np.ndarray,
                        vocal_segments: List[VocalSegment],
                        rms: np.ndarray,
                        sim_matrix: Optional[np.ndarray] = None,
                        ) -> List[StructureSection]:
        """Label each section as intro/verse/chorus/bridge/outro."""
        if not boundaries:
            return [StructureSection(0.0, duration, 'unknown')]

        all_bounds = sorted(set(boundaries + [duration]))
        raw_sections = [(all_bounds[i], all_bounds[i + 1])
                        for i in range(len(all_bounds) - 1)]

        if not raw_sections:
            return [StructureSection(0.0, duration, 'unknown')]

        # Compute features per section
        features = []
        n_beats = int(len(beats))
        n_sim = int(sim_matrix.shape[0]) if sim_matrix is not None else 0
        n = int(min(n_beats, n_sim))

        for start, end in raw_sections:
            sf = int(start * self.sr / self.hop)
            ef = int(end * self.sr / self.hop)
            sec_rms = rms[sf:min(ef, len(rms))]
            avg_energy = float(np.mean(sec_rms)) if len(sec_rms) > 0 else 0.0
            dyn_std = float(np.std(sec_rms)) if len(sec_rms) > 0 else 0.0

            # Vocal ratio
            vocal_dur = sum(
                min(vs.end, end) - max(vs.start, start)
                for vs in vocal_segments
                if vs.start < end and vs.end > start
            )
            vocal_ratio = float(vocal_dur / (end - start)) if end > start else 0.0
            vocal_ratio = float(np.clip(vocal_ratio, 0.0, 1.0))

            # Repetition score: how similar this section is to other parts
            rep_score = 0.0
            if n >= 8:
                bs = int(np.clip(np.searchsorted(beats[:n], start, side='left'), 0, n - 1))
                be = int(np.clip(np.searchsorted(beats[:n], end, side='right'), bs + 1, n))
                if be > bs:
                    idx = np.arange(bs, be, dtype=int)
                    outside = np.concatenate([np.arange(0, bs, dtype=int), np.arange(be, n, dtype=int)])
                    if len(outside) > 0:
                        sim_out = sim_matrix[np.ix_(idx, outside)]
                        raw = float(np.mean(np.max(sim_out, axis=1))) if sim_out.size else 0.0
                        # Scale to 0–1 (empirical): ~0.35 is weak repetition, ~0.80 is strong.
                        rep_score = float(np.clip((raw - 0.35) / 0.45, 0.0, 1.0))

            features.append({
                'energy': avg_energy,
                'dyn_std': dyn_std,
                'vocal_ratio': vocal_ratio,
                'repetition_score': rep_score,
            })

        # Normalize energy and dynamics
        max_e = max((f['energy'] for f in features), default=1e-8) + 1e-8
        max_d = max((f['dyn_std'] for f in features), default=1e-8) + 1e-8
        for f in features:
            f['energy_norm'] = float(f['energy'] / max_e)
            f['dyn_norm'] = float(f['dyn_std'] / max_d)

        # Label using rules
        sections = []
        n = len(raw_sections)
        for i, ((start, end), feat) in enumerate(zip(raw_sections, features)):
            label = 'unknown'

            e = float(feat['energy_norm'])
            vr = float(feat['vocal_ratio'])
            rep = float(feat['repetition_score'])

            if i == 0 and (e < 0.55 or vr < 0.35):
                label = 'intro'
            elif i == n - 1:
                label = 'outro'
            elif e > 0.62 and vr > 0.35 and rep > 0.28:
                label = 'chorus'
            elif e > 0.42 and vr > 0.32:
                label = 'verse'
            elif 0 < i < n - 1 and vr > 0.25 and rep < 0.25:
                label = 'bridge'
            elif vr > 0.20:
                label = 'verse'
            else:
                label = 'instrumental'

            # Boring score (primarily for intros): low energy, low vocals, low dynamics
            boring = (1.0 - e) * 0.55 + (1.0 - vr) * 0.25 + (1.0 - float(feat['dyn_norm'])) * 0.15
            if label == 'intro':
                boring += rep * 0.10
            boring = float(np.clip(boring, 0.0, 1.0))

            sections.append(StructureSection(
                start=start, end=end, label=label,
                energy=float(feat['energy']),
                vocal_ratio=vr,
                repetition_score=rep,
                boring_score=boring,
            ))

        return sections

    # ── Main Section ────────────────────────────────────────────────

    def _identify_main(self, sections: List[StructureSection],
                       beats: np.ndarray,
                       vocal_segments: List[VocalSegment],
                       rms: np.ndarray,
                       sim_matrix: np.ndarray
                       ) -> Optional[StructureSection]:
        """Find the primary chorus (strongest, most repeated section)."""
        candidates = [s for s in sections if s.label in ('chorus', 'verse')]
        if not candidates:
            return None

        max_e = max((float(s.energy) for s in sections), default=1e-8) + 1e-8

        best = None
        best_score = -1.0

        for sec in candidates:
            e = float(sec.energy) / max_e
            rep = float(getattr(sec, 'repetition_score', 0.0) or 0.0)
            score = e * 0.32 + float(sec.vocal_ratio) * 0.22 + rep * 0.36

            if sec.label == 'chorus':
                score += 0.10
            # Duration bonus
            dur = sec.duration
            if 15 <= dur <= 45:
                score += 0.2
            elif dur < 15:
                score += dur / 75.0

            if score > best_score:
                best_score = score
                best = sec

        if best and best_score > 0.3:
            logger.info(
                f"  Main section: {best.label} at {best.start:.1f}–{best.end:.1f}s "
                f"(rep: {float(getattr(best, 'repetition_score', 0.0) or 0.0):.2f})"
            )
        return best
