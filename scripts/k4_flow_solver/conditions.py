"""Numeric geometry only: object/edge tokens, K4 tokens, and task state.

Raster masks in saved scenes are converted to a lossless union of rectangles.
No image features or teacher-derived quantities enter the condition encoder.
The SDF is used only by training losses, never by the numeric model encoder.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import distance_transform_edt
from .config import DEFAULT


def local_points(points, start):
    p = np.asarray(points, dtype=float)-np.asarray(start[:2])
    c, s = np.cos(start[2]), np.sin(start[2])
    return p @ np.array([[c, -s], [s, c]])


def occupancy_rectangles(mask):
    """Exactly partition occupied cells by merging identical horizontal runs."""
    active, rectangles = {}, []
    for r, line in enumerate(np.asarray(mask, dtype=bool)):
        switches = np.diff(np.r_[False, line, False].astype(int))
        runs = set(zip(np.flatnonzero(switches == 1), np.flatnonzero(switches == -1)))
        for run in list(active):
            if run not in runs:
                rectangles.append((active.pop(run), r, *run))
        for run in runs:
            active.setdefault(run, r)
    rectangles.extend((lo, len(mask), *run) for run, lo in active.items())
    return sorted(rectangles)


def scene_tokens(scene, cfg=DEFAULT):
    mask = scene['obstacle_physical']
    height, width = mask.shape
    start = scene['start_state']
    tokens = []

    def token(xy, yaw, length, width, kind):
        row = [xy[0]/cfg.coordinate_scale, xy[1]/cfg.coordinate_scale,
               np.sin(yaw), np.cos(yaw), length/cfg.coordinate_scale, width/cfg.coordinate_scale]
        tokens.append(row+[float(i == kind) for i in range(4)])

    for r0, r1, c0, c1 in occupancy_rectangles(mask):
        center = [10-(r0+r1)*10/height, -10+(c0+c1)*10/width]
        xy = local_points(np.asarray(center)[None], start)[0]
        token(xy, -float(start[2]), (r1-r0)*20/height, (c1-c0)*20/width, 0)
    # Explicit boundary edges avoid representing a free-space polygon as solid.
    for kind, polygon in ((1, scene['parking_lot_polygon_xy']),
                          (2, scene['target_slot_polygon_xy']),
                          (3, np.array([[-10., -10.], [10., -10.], [10., 10.], [-10., 10.]]))):
        polygon = local_points(polygon, start)
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            edge = b-a
            token((a+b)/2, np.arctan2(edge[1], edge[0]), np.linalg.norm(edge), 0.0, kind)
    result = np.asarray(tokens, dtype=np.float32)
    if len(result) > cfg.max_scene_tokens:
        raise ValueError(f'scene_token_overflow:{len(result)}')
    if not np.isfinite(result).all():
        raise ValueError('nonfinite_numeric_scene')
    return result


def path_tokens(path, cfg=DEFAULT):
    p = np.asarray(path, dtype=float)
    if p.ndim != 2 or p.shape[1] != 5 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError('invalid_k4_path')
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))]
    transitions = np.flatnonzero(p[1:, 3] != p[:-1, 3])
    required = np.unique(np.r_[0, transitions, transitions+1, len(p)-1])
    if len(required) > cfg.path_tokens:
        # An oscillating/bad input remains a solver input. Preserve the ends
        # and a bounded sample of cusp points instead of rejecting its quality.
        required = required[np.linspace(0,len(required)-1,cfg.path_tokens).round().astype(int)]
    # Preserve every cusp if it fits; otherwise the token budget is lossy.
    remaining = np.setdiff1d(np.arange(len(p)), required)
    needed = min(cfg.path_tokens-len(required), len(remaining))
    chosen = remaining[np.linspace(0, len(remaining)-1, needed).round().astype(int)] if needed else []
    indices = np.sort(np.r_[required, chosen].astype(int))
    phase = np.r_[0, np.cumsum(p[1:, 3] != p[:-1, 3])]
    a = p[indices]
    return np.c_[a[:, :2]/cfg.coordinate_scale, np.sin(a[:, 2]), np.cos(a[:, 2]),
                 a[:, 4]/cfg.max_steer, a[:, 3], arc[indices]/cfg.coordinate_scale,
                 phase[indices]/12.].astype(np.float32)


def task_state(scene, cfg=DEFAULT):
    start = scene['start_state']
    goal = scene['requested_goal_pose']
    xy = local_points(np.asarray(goal[:2])[None], start)[0]
    yaw = float(goal[2]-start[2])
    return np.asarray([*list(xy/cfg.coordinate_scale), np.sin(yaw), np.cos(yaw),
                       float(start[4])/cfg.max_steer], dtype=np.float32)


def scene_sdf(scene):
    """Meters, positive free; padded occupied cells include observation edges."""
    mask = np.asarray(scene['obstacle_physical'], dtype=bool).copy()
    if mask.shape != (200, 200):
        raise ValueError('saved_scene_raster_contract_requires_200x200')
    x, y = np.meshgrid(10-(np.arange(200)+.5)*.1, -10+(np.arange(200)+.5)*.1, indexing='ij')
    poly = np.asarray(scene['parking_lot_polygon_xy'], dtype=float)
    cross = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        cross.append((b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0]))
    cross = np.stack(cross)
    inside = (cross >= -1e-7).all(axis=0) | (cross <= 1e-7).all(axis=0)
    mask |= ~inside
    padded = np.pad(mask, 1, constant_values=True)
    sdf = (distance_transform_edt(~padded)-distance_transform_edt(padded))*.1
    return sdf[1:-1, 1:-1].astype(np.float32)
