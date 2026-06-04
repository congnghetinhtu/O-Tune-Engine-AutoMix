#!/usr/bin/env python3
"""
OTune Engine v2 — Apple Music-style Automix Engine
===================================================

Modular rewrite optimized for Apple Silicon with frequency-domain
vocal-aware transitions.

Usage:
    python otune.py /path/to/tracks/
    python otune.py /path/to/tracks/ -o mix.wav -c 10
    python otune.py /path/to/tracks/ --no-gpu --threads 1
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from otune_engine.core.config import (
    MixConfig, AnalysisConfig, TransitionConfig,
    AccelerationConfig, CacheConfig
)
from otune_engine.core.pipeline import MixPipeline


def setup_logging(verbose: bool = False):
    """Configure logging with clean formatting."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = '%(message)s' if not verbose else '%(levelname)s [%(name)s] %(message)s'

    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Quiet noisy libraries
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def build_config(args) -> MixConfig:
    """Build MixConfig from CLI arguments."""
    # GPU settings
    use_gpu = args.gpu if args.gpu is not None else not args.no_gpu

    config = MixConfig(
        input_folder=Path(args.input_folder),
        output_file=args.output,
        non_interactive=args.non_interactive,
        start_track_index=(
            args.start_track - 1 if args.start_track and args.start_track > 0
            else None
        ),
        analysis=AnalysisConfig(
            sample_rate=args.sample_rate,
            target_lufs=args.target_lufs,
        ),
        transition=TransitionConfig(
            crossfade_duration=args.crossfade,
            vocal_crossfade_mode=args.vocal_mode,
            vocal_preserve_both=not args.no_preserve_vocals,
            tempo_ramp_max_pct=args.tempo_ramp_max,
            scratch_bend_enabled=not args.no_scratch_bend,
            scratch_bend_max_pct=args.scratch_bend_max,
            beat_align_window_ms=args.beat_align_window_ms,
        ),
        acceleration=AccelerationConfig(
            use_gpu=use_gpu,
            gpu_batch_size=args.gpu_batch,
            max_workers=args.threads,
            use_process_pool=not args.no_process_pool,
        ),
        cache=CacheConfig(
            enabled=not args.no_cache,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        ),
    )
    return config


def main():
    parser = argparse.ArgumentParser(
        description="OTune Engine v2 — Apple Music-style automix with "
                    "vocal-aware transitions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ~/Music/tracks/
  %(prog)s ~/Music/tracks/ -o party_mix.wav -c 10
  %(prog)s ~/Music/tracks/ --vocal-mode duck --threads 8
  %(prog)s ~/Music/tracks/ --no-gpu --start-track 3
        """
    )

    # Required
    parser.add_argument("input_folder",
                        help="Folder containing audio tracks")

    # Output
    parser.add_argument("-o", "--output", default="otune_mix.wav",
                        help="Output filename (default: otune_mix.wav)")

    # Transition
    parser.add_argument("-c", "--crossfade", type=float, default=8.0,
                        help="Crossfade duration in seconds (default: 8.0, max: 30.0)")
    parser.add_argument("--vocal-mode", default="auto",
                        choices=["auto", "duck", "blend", "gap_align"],
                        help="Vocal crossfade strategy (default: auto)")
    parser.add_argument("--no-preserve-vocals", action="store_true",
                        help="Allow vocal ducking/gap alignment (default: preserve both)")
    parser.add_argument("--no-scratch-bend", action="store_true",
                        help="Disable incoming-track scratch bend")
    parser.add_argument("--scratch-bend-max", type=float, default=6.0,
                        help="Max scratch bend percent (default: 6.0)")
    parser.add_argument("--tempo-ramp-max", type=float, default=2.0,
                        help="Max tempo ramp percent during crossfade (default: 2.0)")
    parser.add_argument("--beat-align-window-ms", type=float, default=25.0,
                        help="Beat alignment search window in ms (default: 25.0)")

    # Analysis
    parser.add_argument("-s", "--sample-rate", type=int, default=44100,
                        help="Sample rate (default: 44100)")
    parser.add_argument("--target-lufs", type=float, default=-14.0,
                        help="Target loudness in LUFS (default: -14.0)")

    # Performance
    parser.add_argument("-t", "--threads", type=int, default=3,
                        help="Parallel analysis threads (default: 3)")
    parser.add_argument("--gpu", action="store_true", default=None,
                        help="Force GPU acceleration")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU, CPU only")
    parser.add_argument("--gpu-batch", type=int, default=None,
                        help="GPU batch size (default: auto)")
    parser.add_argument("--no-process-pool", action="store_true",
                        help="Use ThreadPoolExecutor instead of ProcessPoolExecutor")

    # Track selection
    parser.add_argument("--start-track", type=int, default=None,
                        help="Start with track N (1-based)")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Skip interactive track selection (auto-order)")

    # Cache
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable analysis caching")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Custom cache directory")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear cache and exit")

    # Debug
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose debug output")

    args = parser.parse_args()

    # Setup
    setup_logging(args.verbose)
    logger = logging.getLogger('otune')

    # Validate
    if not Path(args.input_folder).is_dir():
        logger.error(f"Input folder not found: {args.input_folder}")
        return 1

    # Build config
    config = build_config(args)

    # Print banner
    logger.info("=" * 60)
    logger.info("  OTune Engine v2")
    logger.info("  Vocal-aware transitions • Apple Silicon optimized")
    logger.info("=" * 60)

    # Handle clear cache
    if args.clear_cache:
        from otune_engine.cache.analysis_cache import AnalysisCache
        cache = AnalysisCache(config.cache_path)
        count = cache.clear()
        logger.info(f"Cleared {count} cache files")
        return 0

    # Run pipeline
    pipeline = MixPipeline(config)

    # Log hardware info
    if pipeline.gpu and pipeline.gpu.use_mps:
        logger.info(f"  GPU: {pipeline.gpu.chip_model} "
                    f"({pipeline.gpu.gpu_cores} cores) ✓")
    else:
        logger.info("  GPU: disabled (CPU mode)")
    logger.info(f"  Workers: {config.acceleration.max_workers}")
    logger.info(f"  Crossfade: {config.transition.crossfade_duration}s")
    logger.info(f"  Vocal mode: {config.transition.vocal_crossfade_mode}")
    logger.info(f"  LUFS target: {config.analysis.target_lufs}")
    logger.info("")

    success = pipeline.run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
