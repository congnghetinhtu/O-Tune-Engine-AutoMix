"""
Parallel processing utilities for CPU-bound analysis.

Uses ProcessPoolExecutor for GIL-free parallelism on Apple Silicon,
which is critical since librosa is CPU-bound and single-threaded.
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Callable, Any, Optional

logger = logging.getLogger(__name__)


def get_optimal_workers(max_requested: int = 4) -> int:
    """Determine optimal worker count based on CPU cores and memory."""
    try:
        cpu_count = os.cpu_count() or 4
        # Leave 1–2 cores free for system responsiveness
        optimal = min(max_requested, max(1, cpu_count - 1))
        return optimal
    except Exception:
        return min(max_requested, 4)


def run_parallel_analysis(
    file_paths: List[Path],
    analyze_fn: Callable[[Path], Any],
    max_workers: int = 4,
    use_processes: bool = False,
) -> List[Any]:
    """
    Analyze multiple audio files in parallel.

    Args:
        file_paths: List of audio file paths.
        analyze_fn: Function that takes a Path and returns analysis result.
        max_workers: Maximum parallel workers.
        use_processes: If True, use ProcessPoolExecutor (GIL-free).
                       If False, use ThreadPoolExecutor (safer for GPU).

    Returns:
        List of analysis results (None entries are filtered out).
    """
    # ProcessPool can't pickle bound methods or objects with thread locks.
    if use_processes and getattr(analyze_fn, '__self__', None) is not None:
        logger.warning("ProcessPool disabled for bound method; using threads")
        use_processes = False

    workers = get_optimal_workers(max_workers)
    logger.info(f"Analyzing {len(file_paths)} tracks with {workers} "
                f"{'process' if use_processes else 'thread'} workers")

    results = []

    # NOTE: ProcessPoolExecutor requires the analyze_fn to be picklable.
    # If using GPU (torch tensors), we must use ThreadPoolExecutor instead,
    # since MPS device objects can't cross process boundaries.
    PoolClass = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    with PoolClass(max_workers=workers) as executor:
        future_to_path = {
            executor.submit(analyze_fn, fp): fp
            for fp in file_paths
        }

        for future in as_completed(future_to_path):
            fp = future_to_path[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze {fp.name}: {e}")

    logger.info(f"Successfully analyzed {len(results)}/{len(file_paths)} tracks")
    return results
