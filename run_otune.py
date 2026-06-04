#!/usr/bin/env python3
"""Minimal example: run OTune Engine on a folder of tracks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from otune_engine.core.config import MixConfig
from otune_engine.core.pipeline import MixPipeline

config = MixConfig(
    input_folder=Path("/path/to/your/tracks"),
    output_file="my_mix.wav",
)
pipeline = MixPipeline(config)
pipeline.run()
