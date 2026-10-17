"""Drop-in exact collision checker with a native distance transform."""

from __future__ import annotations

import math

import numpy as np

from hybrid_astar_8ch_bev import (
    BEV_RESOLUTION,
    BEV_SIZE,
    BEV_X_MAX,
    BEV_Y_MIN,
    CollisionChecker,
)


_GRID_COORDINATES = np.arange(BEV_SIZE, dtype=np.float32)
_SQUARED_GRID_DISTANCE = (
    _GRID_COORDINATES[:, None] - _GRID_COORDINATES[None, :]
) ** 2


def fast_distance_transform(obstacle_mask, resolution=BEV_RESOLUTION):
    """Exact separable squared EDT using vectorized 200x200 reductions."""

    obstacle = np.asarray(obstacle_mask, dtype=bool)
    if obstacle.shape != (BEV_SIZE, BEV_SIZE):
        raise ValueError("obstacle mask must be 200x200")
    if not np.any(obstacle):
        return np.full(obstacle.shape, 1000.0, dtype=np.float32)
    horizontal = np.full(obstacle.shape, np.inf, dtype=np.float32)
    for row in range(BEV_SIZE):
        occupied = np.flatnonzero(obstacle[row])
        if len(occupied):
            horizontal[row] = _SQUARED_GRID_DISTANCE[:, occupied].min(axis=1)
    squared = np.empty_like(horizontal)
    for col in range(BEV_SIZE):
        squared[:, col] = (
            _SQUARED_GRID_DISTANCE + horizontal[:, col][None, :]
        ).min(axis=1)
    return np.sqrt(squared).astype(np.float32) * float(resolution)


class FastCollisionChecker(CollisionChecker):
    """Preserve exact checker semantics while avoiding its Python EDT loops."""

    def __init__(self, obstacle_mask, vehicle):
        self.obstacle = np.asarray(obstacle_mask, dtype=bool)
        if self.obstacle.shape != (BEV_SIZE, BEV_SIZE):
            raise ValueError("obstacle mask must be 200x200")
        self.vehicle = vehicle
        try:
            import cv2

            free = np.asarray(~self.obstacle, dtype=np.uint8)
            self.distance = cv2.distanceTransform(
                free, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            ).astype(np.float32) * float(BEV_RESOLUTION)
            if not np.isfinite(self.distance).all() or float(self.distance.max(initial=0.0)) > 1e6:
                self.distance = fast_distance_transform(self.obstacle, BEV_RESOLUTION)
        except Exception:
            self.distance = fast_distance_transform(self.obstacle, BEV_RESOLUTION)
        self.circle_offsets, self.circle_radius = vehicle.covering_circles()
        self.cell_half_diagonal = (
            math.sqrt(2.0)
            * 0.5
            * BEV_RESOLUTION
            * max(0.0, float(vehicle.raster_cell_margin_scale))
        )
        obstacle_rows, obstacle_cols = np.nonzero(self.obstacle)
        self.obstacle_x = BEV_X_MAX - (obstacle_rows.astype(np.float64) + 0.5) * BEV_RESOLUTION
        self.obstacle_y = BEV_Y_MIN + (obstacle_cols.astype(np.float64) + 0.5) * BEV_RESOLUTION


class ExactRasterFootprintCollisionChecker:
    """Exact Tesla OBB versus occupied 0.1 m raster-cell validation.

    HOPE already rolls its physical vehicle polygon against rectangles made
    from the same occupied cells.  The general Hybrid A* checker adds a 0.18 m
    tracking reserve, which double-counts clearance when used as a second gate
    on a HOPE path.  This checker keeps the independent raster validation but
    applies no hidden body inflation.  Collision with the physical vehicle or
    the BEV boundary is still rejected.
    """

    def __init__(self, obstacle_mask, vehicle, footprint_margin=0.0):
        self.obstacle = np.asarray(obstacle_mask, dtype=bool)
        if self.obstacle.shape != (BEV_SIZE, BEV_SIZE):
            raise ValueError("obstacle mask must be 200x200")
        self.vehicle = vehicle
        self.footprint_margin = max(0.0, float(footprint_margin))
        rows, cols = np.nonzero(self.obstacle)
        self.obstacle_x = (
            BEV_X_MAX - (rows.astype(np.float64) + 0.5) * BEV_RESOLUTION
        )
        self.obstacle_y = (
            BEV_Y_MIN + (cols.astype(np.float64) + 0.5) * BEV_RESOLUTION
        )
        self.cell_half = 0.5 * float(BEV_RESOLUTION)

    def check_poses(self, xs, ys, yaws):
        x = np.asarray(xs, dtype=np.float64).reshape((-1,))
        y = np.asarray(ys, dtype=np.float64).reshape((-1,))
        yaw = np.asarray(yaws, dtype=np.float64).reshape((-1,))
        if not (len(x) == len(y) == len(yaw)) or len(x) == 0:
            return False, -float("inf")

        half_l = 0.5 * float(self.vehicle.length) + self.footprint_margin
        half_w = 0.5 * float(self.vehicle.width) + self.footprint_margin
        offset_x = float(self.vehicle.rear_axle_to_center)
        offset_y = float(self.vehicle.rear_axle_lateral_offset)
        minimum_separation = float("inf")

        for px, py, heading in zip(x, y, yaw):
            c = math.cos(float(heading))
            s = math.sin(float(heading))
            center_x = float(px) + offset_x * c - offset_y * s
            center_y = float(py) + offset_x * s + offset_y * c

            extent_x = half_l * abs(c) + half_w * abs(s)
            extent_y = half_l * abs(s) + half_w * abs(c)
            if (
                center_x - extent_x < -BEV_X_MAX
                or center_x + extent_x > BEV_X_MAX
                or center_y - extent_y < BEV_Y_MIN
                or center_y + extent_y > -BEV_Y_MIN
            ):
                return False, -float("inf")
            if len(self.obstacle_x) == 0:
                continue

            dx = self.obstacle_x - center_x
            dy = self.obstacle_y - center_y
            local_x = dx * c + dy * s
            local_y = -dx * s + dy * c
            cell_projection = self.cell_half * (abs(c) + abs(s))

            # Separating axes for an oriented vehicle rectangle and an
            # axis-aligned occupied cell: vehicle x/y and world x/y.
            gaps = np.stack(
                (
                    np.abs(local_x) - (half_l + cell_projection),
                    np.abs(local_y) - (half_w + cell_projection),
                    np.abs(dx) - (self.cell_half + extent_x),
                    np.abs(dy) - (self.cell_half + extent_y),
                ),
                axis=0,
            )
            pair_separation = np.max(gaps, axis=0)
            # Treat numerical contact as collision. Grid coordinates such as
            # 1.05 are not exactly representable in binary floating point.
            if np.any(pair_separation <= 1.0e-9):
                return False, float(pair_separation.min(initial=0.0))
            minimum_separation = min(
                minimum_separation,
                float(pair_separation.min(initial=float("inf"))),
            )

        if not np.isfinite(minimum_separation):
            minimum_separation = 1000.0
        return True, float(minimum_separation)

    def pose_ok(self, pose):
        return self.check_poses([pose[0]], [pose[1]], [pose[2]])
