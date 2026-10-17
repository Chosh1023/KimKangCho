"""Static OGM hybrid reinforcement-learning parking planner.

The package deliberately keeps CARLA/Pygame imports out of the policy worker.
Geometry, observations, training, and inference therefore share exactly the
same rear-axle coordinate contract in both Python 3.7 and Python 3.11.
"""

from .config import EnvConfig, GridConfig, RewardConfig, SACConfig, ShieldConfig
from .contract import OGM_CHANNELS, StaticOGM, StaticScenario
from .kinematics import KinematicState, integrate_action

__all__ = [
    "EnvConfig",
    "GridConfig",
    "RewardConfig",
    "SACConfig",
    "ShieldConfig",
    "OGM_CHANNELS",
    "StaticOGM",
    "StaticScenario",
    "KinematicState",
    "integrate_action",
]
