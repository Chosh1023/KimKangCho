"""Trackable rear-axle bicycle dynamics shared by environment and planner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from hybrid_astar_8ch_bev import VehicleConfig, wrap_pi

from .config import ShieldConfig


@dataclass(frozen=True)
class KinematicState:
    x: float
    y: float
    yaw: float
    steer: float = 0.0
    direction: int = 1

    @property
    def pose(self) -> Tuple[float, float, float]:
        return float(self.x), float(self.y), float(self.yaw)

    @classmethod
    def from_sequence(cls, value: Sequence[float]) -> "KinematicState":
        return cls(
            x=float(value[0]),
            y=float(value[1]),
            yaw=wrap_pi(float(value[2])),
            steer=float(value[3]) if len(value) >= 4 else 0.0,
            direction=-1 if len(value) >= 5 and float(value[4]) < 0.0 else 1,
        )


def constant_curvature_pose(
    x: np.ndarray,
    y: np.ndarray,
    yaw: np.ndarray,
    steer: np.ndarray,
    signed_distance: np.ndarray,
    wheel_base: float,
):
    """Vectorized exact integration for constant steering and arc length."""

    curvature = np.tan(steer) / max(1e-9, float(wheel_base))
    next_yaw = yaw + curvature * signed_distance
    straight = np.abs(curvature) < 1e-9
    safe_curvature = np.where(straight, 1.0, curvature)
    next_x = x + (np.sin(next_yaw) - np.sin(yaw)) / safe_curvature
    next_y = y + (-np.cos(next_yaw) + np.cos(yaw)) / safe_curvature
    next_x = np.where(straight, x + signed_distance * np.cos(yaw), next_x)
    next_y = np.where(straight, y + signed_distance * np.sin(yaw), next_y)
    return next_x, next_y, (next_yaw + math.pi) % (2.0 * math.pi) - math.pi


def integrate_action(
    state: KinematicState,
    target_steer_rad: float,
    speed_mps: float,
    vehicle: VehicleConfig,
    config: ShieldConfig,
) -> Tuple[KinematicState, np.ndarray]:
    """Integrate one decision while enforcing the real steering-rate limit.

    Returned rows follow the live follower contract
    ``[rear_x, rear_y, yaw, direction, bicycle_steer]``.
    """

    target = float(np.clip(target_steer_rad, -vehicle.max_steer, vehicle.max_steer))
    speed = float(speed_mps)
    substeps = max(1, int(math.ceil(config.decision_dt_s / 0.05)))
    dt = config.decision_dt_s / float(substeps)
    max_delta = math.radians(config.steering_rate_deg_s) * dt
    x, y, yaw, steer = float(state.x), float(state.y), float(state.yaw), float(state.steer)
    direction = state.direction if abs(speed) < 1e-8 else (1 if speed > 0.0 else -1)
    rows = [[x, y, yaw, float(direction), steer]]
    for _ in range(substeps):
        steer += float(np.clip(target - steer, -max_delta, max_delta))
        ds = speed * dt
        nx, ny, nyaw = constant_curvature_pose(
            np.asarray(x),
            np.asarray(y),
            np.asarray(yaw),
            np.asarray(steer),
            np.asarray(ds),
            vehicle.wheel_base,
        )
        x, y, yaw = float(nx), float(ny), float(nyaw)
        rows.append([x, y, yaw, float(direction), steer])
    return KinematicState(x, y, yaw, steer, direction), np.asarray(rows, dtype=np.float64)
