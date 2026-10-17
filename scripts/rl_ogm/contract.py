"""Validated high-resolution OGM and planner observation contracts."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from hybrid_astar_8ch_bev import VehicleConfig, wrap_pi

from .config import GridConfig


OGM_CHANNELS = ("occupancy",)
OGM_FREE = np.uint8(0)
OGM_UNKNOWN = np.uint8(127)
OGM_OCCUPIED = np.uint8(255)


def _binary(name: str, value: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError("{} must have shape {}, got {}".format(name, shape, array.shape))
    if not np.isfinite(array).all():
        raise ValueError("{} contains non-finite values".format(name))
    return np.asarray(array > 0, dtype=np.uint8)


@dataclass(frozen=True)
class StaticOGM:
    """Paper-style one-channel OGM in the fixed initial-ego planning frame.

    The policy sees only occupancy: 0=free, 127=unknown and 255=occupied.
    Safety dilation stays outside the neural observation and is retained only
    for exact collision checking and the swept-volume action shield.
    """

    raster: np.ndarray
    grid: GridConfig = GridConfig()
    safety_mask: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.grid.validate()
        array = np.asarray(self.raster)
        expected = (len(OGM_CHANNELS), self.grid.height, self.grid.width)
        if array.shape != expected:
            raise ValueError("OGM must have shape {}, got {}".format(expected, array.shape))
        if array.dtype != np.uint8:
            raise ValueError("OGM raster must be uint8")
        values = np.unique(array)
        allowed = np.asarray([OGM_FREE, OGM_UNKNOWN, OGM_OCCUPIED], dtype=np.uint8)
        if not np.isin(values, allowed).all():
            raise ValueError("OGM values must be 0=free, 127=unknown, or 255=occupied")
        object.__setattr__(self, "raster", np.ascontiguousarray(array))
        safety = self.physical if self.safety_mask is None else _binary(
            "safety_mask", self.safety_mask, (self.grid.height, self.grid.width)
        ).astype(bool)
        safety = np.logical_or(safety, self.physical)
        object.__setattr__(self, "safety_mask", np.ascontiguousarray(safety))

    @property
    def physical(self) -> np.ndarray:
        return self.raster[0] == OGM_OCCUPIED

    @property
    def safety(self) -> np.ndarray:
        return self.safety_mask

    @property
    def observed(self) -> np.ndarray:
        return self.raster[0] != OGM_UNKNOWN

    @property
    def hard_safety(self) -> np.ndarray:
        return np.logical_or(self.safety, np.logical_not(self.observed))

    @property
    def model_input(self) -> np.ndarray:
        return self.raster.astype(np.float32) / 255.0

    @property
    def content_hash(self) -> str:
        return hashlib.blake2b(self.raster.tobytes(), digest_size=16).hexdigest()

    @classmethod
    def from_layers(
        cls,
        physical_occupancy: np.ndarray,
        safety_occupancy: Optional[np.ndarray] = None,
        observed_space: Optional[np.ndarray] = None,
        grid: GridConfig = GridConfig(),
    ) -> "StaticOGM":
        grid.validate()
        shape = (grid.height, grid.width)
        physical = _binary("physical_occupancy", physical_occupancy, shape)
        safety = physical.copy() if safety_occupancy is None else _binary("safety_occupancy", safety_occupancy, shape)
        safety = np.maximum(safety, physical)
        observed = np.ones(shape, dtype=np.uint8) if observed_space is None else _binary("observed_space", observed_space, shape)
        occupancy = np.where(
            physical > 0,
            OGM_OCCUPIED,
            np.where(observed > 0, OGM_FREE, OGM_UNKNOWN),
        ).astype(np.uint8)
        return cls(
            np.ascontiguousarray(occupancy[None]),
            grid=grid,
            safety_mask=np.ascontiguousarray(safety > 0),
        )


@dataclass(frozen=True)
class StaticScenario:
    scenario_id: str
    ogm: StaticOGM
    start_rear_pose: Tuple[float, float, float]
    goal_rear_pose: Tuple[float, float, float]
    goal_polygon: Tuple[Tuple[float, float], ...] = ()
    start_steer_rad: float = 0.0
    metadata: Optional[Mapping[str, object]] = None

    def __post_init__(self) -> None:
        for name, pose in (("start", self.start_rear_pose), ("goal", self.goal_rear_pose)):
            if len(pose) < 3 or not np.isfinite(np.asarray(pose[:3], dtype=np.float64)).all():
                raise ValueError("{} pose must be finite [x,y,yaw]".format(name))


def state_vector(
    rear_pose: Sequence[float],
    steer_rad: float,
    goal_rear_pose: Sequence[float],
    vehicle: VehicleConfig,
    last_speed_fraction: float = 0.0,
    step_fraction: float = 0.0,
    grid: GridConfig = GridConfig(),
) -> np.ndarray:
    """Return the 16-D state used by both the actor and both critics."""

    x, y, yaw = map(float, rear_pose[:3])
    gx, gy, gyaw = map(float, goal_rear_pose[:3])
    dx = gx - x
    dy = gy - y
    c = math.cos(yaw)
    s = math.sin(yaw)
    body_x = c * dx + s * dy
    body_y = -s * dx + c * dy
    distance = math.hypot(dx, dy)
    if distance > 1e-9:
        bearing_cos = body_x / distance
        bearing_sin = body_y / distance
    else:
        bearing_cos, bearing_sin = 1.0, 0.0
    yaw_error = wrap_pi(gyaw - yaw)
    scale_x = max(abs(grid.x_min), abs(grid.x_max), 1.0)
    scale_y = max(abs(grid.y_min), abs(grid.y_max), 1.0)
    vector = np.asarray(
        [
            x / scale_x,
            y / scale_y,
            math.cos(yaw),
            math.sin(yaw),
            float(steer_rad) / max(1e-6, float(vehicle.max_steer)),
            body_x / (2.0 * scale_x),
            body_y / (2.0 * scale_y),
            distance / math.hypot(2.0 * scale_x, 2.0 * scale_y),
            bearing_cos,
            bearing_sin,
            math.cos(yaw_error),
            math.sin(yaw_error),
            gx / scale_x,
            gy / scale_y,
            float(np.clip(last_speed_fraction, -1.0, 1.0)),
            float(np.clip(step_fraction, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )
    return vector


def metric_to_cell(x: float, y: float, grid: GridConfig = GridConfig()) -> Tuple[int, int]:
    row = int(math.floor((grid.x_max - float(x)) / grid.resolution))
    col = int(math.floor((float(y) - grid.y_min) / grid.resolution))
    return row, col
