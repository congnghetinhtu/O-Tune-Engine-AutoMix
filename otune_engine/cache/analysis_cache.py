"""
Thread-safe analysis result caching.

Stores analysis results as JSON (no pickle) for portability.
Audio data is never cached — too large and loaded fresh each time.
"""

import hashlib
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


class AnalysisCache:
    """Thread-safe JSON-based analysis cache."""

    def __init__(self, cache_dir: Path, version: str = '2.0'):
        self.cache_dir = cache_dir
        self.version = version
        self._lock = threading.Lock()

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._cleanup_stale()

    # ── Public API ──────────────────────────────────────────────────

    def load(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load cached analysis for a file. Returns None if miss."""
        cache_path = self._cache_path(file_path)
        if not cache_path.exists():
            return None

        try:
            with self._lock:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            if data.get('cache_version') != self.version:
                return None
            if data.get('file_hash') != self._file_hash(file_path):
                return None

            analysis = data['analysis']

            # Restore numpy arrays from lists
            for key in ('beats', 'downbeats', 'beat_strengths', 'beat_confidence',
                        'mfcc_mean', 'beat_frames'):
                if key in analysis and analysis[key] is not None:
                    analysis[key] = np.array(analysis[key])

            # Restore Path
            if 'file_path' in analysis:
                analysis['file_path'] = Path(analysis['file_path'])

            logger.info(f"  ✓ Cache hit: {file_path.name}")
            return analysis

        except Exception as e:
            logger.debug(f"Cache load failed for {file_path.name}: {e}")
            return None

    def save(self, file_path: Path, analysis: Dict[str, Any]) -> None:
        """Save analysis results to cache."""
        try:
            cache_path = self._cache_path(file_path)

            cache_data = {
                'cache_version': self.version,
                'file_hash': self._file_hash(file_path),
                'cached_at': datetime.now().isoformat(),
                'analysis': {}
            }

            for key, value in analysis.items():
                # Skip audio data — too large
                if key in ('audio_data',):
                    continue
                elif key == 'file_path':
                    cache_data['analysis'][key] = str(value)
                elif isinstance(value, np.ndarray):
                    cache_data['analysis'][key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    cache_data['analysis'][key] = float(value)
                elif isinstance(value, list) and value and isinstance(value[0], dict):
                    # Serialize dataclass-like dicts
                    cache_data['analysis'][key] = value
                elif isinstance(value, list) and value and isinstance(value[0], tuple):
                    cache_data['analysis'][key] = [list(item) for item in value]
                else:
                    cache_data['analysis'][key] = value

            with self._lock:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2, default=str)

            logger.info(f"  ✓ Cached: {file_path.name}")

        except Exception as e:
            logger.warning(f"Cache save failed for {file_path.name}: {e}")

    def clear(self) -> int:
        """Clear all cache files. Returns count deleted."""
        with self._lock:
            count = 0
            for f in self.cache_dir.glob('*.json'):
                f.unlink()
                count += 1
            logger.info(f"Cleared {count} cache files")
            return count

    # ── Internal ────────────────────────────────────────────────────

    def _file_hash(self, file_path: Path) -> str:
        """Content-based hash (size + mtime + first 8KB)."""
        stat = file_path.stat()
        hash_input = f"{stat.st_size}_{stat.st_mtime_ns}".encode()
        try:
            with open(file_path, 'rb') as f:
                hash_input += f.read(8192)
        except Exception:
            pass
        return hashlib.md5(hash_input).hexdigest()

    def _cache_path(self, file_path: Path) -> Path:
        """Generate cache file path."""
        fhash = self._file_hash(file_path)
        safe_name = "".join(c for c in file_path.stem
                            if c.isalnum() or c in (' ', '-', '_'))[:50]
        return self.cache_dir / f"{safe_name}_{fhash}.json"

    def _cleanup_stale(self) -> None:
        """Remove cache files for songs that no longer exist."""
        removed = 0
        for f in self.cache_dir.glob('*.json'):
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                path_str = data.get('analysis', {}).get('file_path')
                if not path_str or not Path(path_str).exists():
                    f.unlink(missing_ok=True)
                    removed += 1
            except Exception:
                # If a cache file is corrupted, leave it to be replaced lazily.
                continue
        if removed > 0:
            logger.info(f"Removed {removed} stale cache files")
