"""One-shot HOPE trajectory distillation planner.

The package is intentionally independent from the Pygame environment at
training and inference time.  Raw collector shards are converted to a stable
semantic-BEV/vector contract by :mod:`dataset`, then consumed by the
coarse-to-fine planner in :mod:`model`.
"""

from .config import HopePathConfig, load_config, save_config
from .model import HopePathPlanner

__all__ = ["HopePathConfig", "HopePathPlanner", "load_config", "save_config"]
