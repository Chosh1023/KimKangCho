"""Warm-start paths for the Flow solver: Hybrid A* planning and frame conversion.

Frame contract (identical to the frozen-K4 cache consumed by ``path_tokens``):

* rows are ``[x, y, yaw, gear(+1/-1), steer]``;
* poses are the rear axle, expressed in the start-local frame, so the first
  row is ``(0, 0, 0)``;
* the scene/BEV frame is x+ ego-forward, y+ ego-right.  The start-local frame
  is only a translation+rotation of it (no reflection), so y+ stays
  ego-right and steering keeps its sign (positive steer increases yaw).

The planner is ``hybrid_astar_8ch_bev.HybridAStarPlanner`` (Hybrid A* with
periodic Reeds-Shepp analytic expansion near the goal), the same planner the
Li iterative OCP used for its guiding route.  Its default vehicle is checked
against ``SolverConfig`` so the path respects the OCP steering limit and
wheelbase.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import math
import time
import numpy as np

from .config import DEFAULT, SolverConfig
from .conditions import local_points

SCHEMA = 'warmstart-path-cache-v1'
# Agreed 2026-09-25: the search stops on an expansion count, not the wall
# clock, so a scene yields the same warm start on any machine or load.  The
# time limit is only a safety stop.  Training caches and experiment 1 must use
# the same values.
AGREED_PLANNER_OVERRIDES = dict(max_expansions=3000, max_time_s=600.0)
FRAME = 'start-local rear axle; x+ forward, y+ ego-right (scene frame rotated, no reflection)'


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def scene_to_local(states, start):
    """Scene-frame rear-axle rows [x, y, yaw, gear, steer] -> start-local rows."""
    s = np.asarray(states, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] < 5 or len(s) < 2:
        raise ValueError('invalid_path_shape')
    out = np.empty((len(s), 5), dtype=np.float64)
    out[:, :2] = local_points(s[:, :2], start)
    out[:, 2] = np.unwrap(wrap(s[:, 2]-float(start[2])))
    out[:, 3] = np.where(s[:, 3] < 0, -1.0, 1.0)
    out[:, 4] = s[:, 4]
    return normalize_gears(out)


def normalize_gears(path):
    """Label each pose with the gear of the motion that reaches it.

    Planner start nodes carry the requested start direction even when the
    first primitive moves the other way; ``path_tokens`` would then see a
    spurious cusp at index 0.  The first row takes the gear of the first move.
    """
    p = np.asarray(path, dtype=np.float64).copy()
    if len(p) > 1:
        p[0, 3] = p[1, 3]
    return p


def insert_cusp_rows(path, cfg=DEFAULT):
    """Give every new gear phase its own first row at the cusp pose.

    A planner row stores the steer *reached* at that pose, so the cusp row
    carries the previous phase's final steer.  ``from_k4`` would then blend
    that value into the first interval of the next phase (Hybrid A* actually
    releases steering to 0 at a cusp; Reeds-Shepp segments start at their own
    constant steer).  The duplicate row has the new gear and a first steer
    chosen so the linear ramp over the first interval reproduces that
    interval's measured curvature.  Geometry is unchanged; ``split_path``
    drops the zero-length cusp interval, and ``path_tokens`` still sees one
    transition.  Returns (path, number of inserted rows).
    """
    p = np.asarray(path, dtype=np.float64)
    out, inserted = [p[0]], 0
    for i in range(1, len(p)):
        out.append(p[i])
        if i+1 < len(p) and p[i+1, 3] != p[i, 3]:
            ds = float(np.hypot(*(p[i+1, :2]-p[i, :2])))
            if ds < 1e-9:
                continue
            curvature = float(wrap(p[i+1, 2]-p[i, 2]))/(p[i+1, 3]*ds)
            mid = math.atan(cfg.wheelbase*curvature)
            row = p[i].copy()
            row[3] = p[i+1, 3]
            row[4] = float(np.clip(2*mid-p[i+1, 4], -cfg.max_steer, cfg.max_steer))
            out.append(row)
            inserted += 1
    return np.asarray(out), inserted


def path_phases(path, cfg=DEFAULT):
    """Gear sequence exactly as ``numerical.from_k4`` will split it."""
    from fast_multiphase_ocp import split_path
    parts, gears = split_path(path)
    lengths = [float(s[-1]) for _, s in parts]
    return [int(g) for g in gears], lengths


def planning_mask(scene):
    """Obstacles plus everything outside the parking-lot polygon.

    The validator rejects any footprint corner outside the lot polygon, so the
    planner receives the same information the Flow model sees as lot edges.
    """
    mask = np.asarray(scene['obstacle_physical'], dtype=bool).copy()
    if mask.shape != (200, 200):
        raise ValueError('saved_scene_raster_contract_requires_200x200')
    poly = np.asarray(scene['parking_lot_polygon_xy'], dtype=float).reshape(-1, 2)
    if len(poly) >= 3:
        x, y = np.meshgrid(10-(np.arange(200)+.5)*.1, -10+(np.arange(200)+.5)*.1, indexing='ij')
        cross = np.stack([(b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0])
                          for a, b in zip(poly, np.roll(poly, -1, axis=0))])
        inside = (cross >= -1e-7).all(axis=0) | (cross <= 1e-7).all(axis=0)
        mask |= ~inside
    return mask


def default_vehicle(cfg=DEFAULT):
    from hybrid_astar_8ch_bev import VehicleConfig
    vehicle = VehicleConfig()
    if abs(vehicle.max_steer-cfg.max_steer) > 1e-9 or abs(vehicle.wheel_base-cfg.wheelbase) > 1e-9:
        raise ValueError('hybrid_astar_vehicle_mismatch_with_solver_config')
    if abs(vehicle.safety_margin-cfg.clearance) > 1e-9:
        raise ValueError('hybrid_astar_margin_mismatch_with_solver_clearance')
    return vehicle


def planner_config(**overrides):
    from hybrid_astar_8ch_bev import PlannerConfig
    return replace(PlannerConfig(), **{k: v for k, v in overrides.items() if v is not None})


def plan_hybrid_astar(scene, planner_cfg, *, vehicle=None, start_direction=1, cfg=DEFAULT):
    """Plan in the scene frame and return (local_path or None, info dict)."""
    from hybrid_astar_8ch_bev import HybridAStarPlanner
    vehicle = vehicle or default_vehicle(cfg)
    start = np.asarray(scene['start_state'], dtype=float)[:3]
    goal = np.asarray(scene['requested_goal_pose'], dtype=float)[:3]
    tick = time.perf_counter()
    planner = HybridAStarPlanner(vehicle, planner_cfg, planning_mask(scene))
    setup = time.perf_counter()-tick
    result = planner.plan(start, goal, start_direction=start_direction)
    total = time.perf_counter()-tick
    info = dict(success=bool(result.success), message=str(result.message), plan_time_s=total,
                setup_time_s=setup, search_time_s=float(result.elapsed_s),
                expansions=int(result.expansions), analytic_expansions=int(result.analytic_expansions),
                min_clearance_m=float(result.min_clearance_m), start_direction=int(start_direction))
    states = np.asarray(result.states, dtype=np.float64)
    if not result.success or len(states) < 2:
        return None, info
    end = states[-1, :3]
    info['terminal_error_scene'] = [float(end[0]-goal[0]), float(end[1]-goal[1]), float(wrap(end[2]-goal[2]))]
    path, info['cusp_rows_inserted'] = insert_cusp_rows(scene_to_local(states, start), cfg)
    return path, info


def known_route_path(raw_scene, cfg=DEFAULT):
    """Convert a saved ``known_feasible_route`` (scene frame) to the cache frame.

    Saved routes start at x=-1.41564 (the 8ch vehicle's rear-axle offset),
    2.92 cm behind ``start_state``; the offset is kept and reported, not hidden.
    Returns (path, info).
    """
    if 'known_feasible_route' not in raw_scene:
        raise ValueError('no_known_feasible_route')
    route = np.asarray(raw_scene['known_feasible_route'], dtype=np.float64)
    if route.ndim != 2 or route.shape[1] < 5:
        raise ValueError(f'unsupported_known_route_shape:{route.shape}')
    local = scene_to_local(route[:, :5], np.asarray(raw_scene['start_state'], dtype=float))
    path, inserted = insert_cusp_rows(local, cfg)
    return path, dict(cusp_rows_inserted=inserted, start_offset_local=[float(v) for v in local[0, :3]])


def planner_metadata(planner_cfg, vehicle, cfg=DEFAULT):
    return dict(planner='hybrid_astar_8ch_bev.HybridAStarPlanner.plan',
                planner_config=asdict(planner_cfg),
                vehicle=dict(max_steer=float(vehicle.max_steer), wheel_base=float(vehicle.wheel_base),
                             safety_margin=float(vehicle.safety_margin),
                             rear_axle_to_center=float(vehicle.rear_axle_to_center),
                             length=float(vehicle.length), width=float(vehicle.width)),
                solver_config_match=dict(max_steer=cfg.max_steer, wheelbase=cfg.wheelbase, clearance=cfg.clearance),
                obstacle_mask='obstacle_physical | outside parking_lot_polygon_xy',
                goal='requested_goal_pose (goal_state and known_feasible_route are not read)')


# ---------------------------------------------------------------------------
# Stage 2: warm-start path -> numerical Controls (reads the stage-1 cache as is)
# ---------------------------------------------------------------------------

STEERING_READING = 'geometric_interval_curvature'
STEER_LIMIT_TOLERANCE = 1e-9


def load_warm_start(file):
    """Return (path, meta) from a stage-1 cache file without modifying it."""
    import json
    with np.load(file, allow_pickle=False) as z:
        return z['path'].astype(np.float64), json.loads(str(z['meta']))


def strip_cusp_rows(path):
    """Drop the zero-length rows ``insert_cusp_rows`` added at gear changes."""
    p = np.asarray(path, dtype=np.float64)
    keep = np.ones(len(p), dtype=bool)
    moved = np.hypot(*np.diff(p[:, :2], axis=0).T) > 1e-12
    flipped = p[1:, 3] != p[:-1, 3]
    keep[1:] = moved | ~flipped
    return p[keep]


def path_intervals(path):
    """Per moving interval: gear, arc length and signed curvature from geometry.

    Gear phases come from ``fast_multiphase_ocp.split_path`` - the exact rule
    the OCP (experiment 2, new x1 targets) applies to the same warm start:
    direction from geometry, and gear runs shorter than 1 cm merged into a
    neighbour.  Flow x0 and IPOPT therefore always start from one structure.
    Every 8ch Hybrid A* sub-step (0.05 m) and Reeds-Shepp sample interval is a
    constant-curvature arc, so curvature = heading change / signed arc length
    is exact and independent of the planner's steer-column labels.  Inside a
    merged sub-centimetre run the motion opposes the phase gear; its curvature
    is taken with the phase gear (< 1 cm of path, same as the OCP seed).
    """
    from fast_multiphase_ocp import split_path
    parts, gears = split_path(np.asarray(path, dtype=np.float64))
    gear, arc, curvature = [], [], []
    for (p, _), g in zip(parts, gears):
        chord = np.hypot(*np.diff(p[:, :2], axis=0).T)
        dtheta = np.diff(p[:, 2])          # split_path unwrapped the heading
        half = 0.5*dtheta
        safe = np.where(np.abs(half) > 1e-12, half, 1.0)
        a = np.where(np.abs(half) > 1e-12, chord*half/np.sin(safe), chord)
        gear.append(np.full(len(a), float(g)))
        arc.append(a)
        curvature.append(dtheta/(g*a))
    return np.concatenate(gear), np.concatenate(arc), np.concatenate(curvature)


def canonical_gear_labels(path):
    """Relabel the gear column with the ``split_path`` phases (for path tokens).

    Rows keep their poses and steer labels; each row takes the gear of the
    moving interval that reaches it, and zero-length rows (start node, cusp
    rows) take the gear of the next moving interval.  Returns (path, changed).
    """
    from fast_multiphase_ocp import split_path
    p = np.asarray(path, dtype=np.float64).copy()
    _, gears = split_path(p)
    ds = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    moving = np.flatnonzero(ds > 1e-6)
    interval_gear = np.zeros(len(ds))
    for (start, stop), g in zip(_phase_runs(p, moving, gears), gears):
        interval_gear[moving[start:stop]] = g
    labels = np.empty(len(p))
    following = float(gears[-1])
    for i in range(len(ds)-1, -1, -1):      # row i+1 is reached by interval i
        if ds[i] > 1e-6:
            following = interval_gear[i]
        labels[i+1] = following             # zero-length row: gear of the next motion
    labels[0] = gears[0]
    changed = int(np.sum(labels != np.where(p[:, 3] < 0, -1.0, 1.0)))
    p[:, 3] = labels
    return p, changed


def _phase_runs(p, moving, gears):
    """Index ranges into ``moving`` for each split_path phase (same merge rule)."""
    theta = np.unwrap(p[:, 2])
    mid = (theta[:-1]+theta[1:])/2
    dot = np.sum(np.diff(p[:, :2], axis=0)*np.c_[np.cos(mid), np.sin(mid)], axis=1)
    ds = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    signs = np.where(dot[moving] >= 0, 1, -1)
    while True:
        runs = np.split(np.arange(len(moving)), np.flatnonzero(signs[1:] != signs[:-1])+1)
        short = next((i for i, r in enumerate(runs) if ds[moving[r]].sum() < .01), None)
        if short is None or len(runs) == 1:
            break
        neighbour = short-1 if short else 1
        signs[runs[short]] = signs[runs[neighbour][0]]
    runs = np.split(np.arange(len(moving)), np.flatnonzero(signs[1:] != signs[:-1])+1)
    if [int(signs[r[0]]) for r in runs] != [int(g) for g in gears]:
        raise ValueError('gear_relabel_disagrees_with_split_path')
    return [(int(r[0]), int(r[-1])+1) for r in runs]


def controls_from_hybrid_astar(path, wheelbase_m=None, cfg=DEFAULT):
    """Hybrid A* warm start -> retimed ``Controls`` on the solver grid.

    Steering is piecewise constant per planner interval: every cosine-grid knot
    takes ``atan(L_solver * curvature)`` of the interval containing its arc
    position (no half-interval shift from interpolating row labels).  Measuring
    curvature from geometry already preserves it across wheelbases
    (delta' = atan(L_solver tan(delta) / L_planner)); a result beyond
    ``max_steer`` is rejected, never clipped.  Speed starts at the limit with
    zero end speeds and is retimed.  Experiment 4 (Hybrid A* alone) costs
    this output.  Returns (Controls, info).
    """
    from .numerical import Controls, grid, retime
    gear, arc, curvature = path_intervals(path)
    if len(arc) == 0:
        raise ValueError('warm_start_has_no_motion')
    delta = np.arctan(cfg.wheelbase*curvature)
    worst = float(np.abs(delta).max())
    if worst > cfg.max_steer+STEER_LIMIT_TOLERANCE:
        raise ValueError(f'steer_exceeds_max_after_curvature_conversion:{worst:.6f}')
    delta = np.clip(delta, -cfg.max_steer, cfg.max_steer)
    cuts = np.flatnonzero(gear[1:] != gear[:-1])+1
    groups = np.split(np.arange(len(arc)), cuts)
    if len(groups) > cfg.max_phases:
        raise ValueError(f'phase_count_out_of_scope:{len(groups)}')
    lengths, knots = [], []
    for g in groups:
        edges = np.r_[0.0, np.cumsum(arc[g])]
        s = grid(cfg.intervals)*edges[-1]
        index = np.clip(np.searchsorted(edges, s, side='right')-1, 0, len(g)-1)
        lengths.append(edges[-1])
        knots.append(delta[g][index])
    knots = np.asarray(knots)
    b = np.full_like(knots, (cfg.max_speed-cfg.control_margin)**2)
    b[:, [0, -1]] = 0
    gears = np.asarray([int(gear[g[0]]) for g in groups])
    controls = retime(Controls(np.asarray(lengths), knots, b, gears), cfg)
    info = dict(steering_reading=STEERING_READING, intervals=int(len(arc)), phases=len(groups),
                gears=gears.tolist(), max_abs_steer=worst,
                curvature_conversion=bool(wheelbase_m is not None and abs(float(wheelbase_m)-cfg.wheelbase) > 1e-9))
    return controls, info


def max_deviation(points, polyline):
    """Largest distance from ``points`` to a dense polyline (point-to-segment)."""
    pts = np.asarray(points, dtype=np.float64)[:, :2]
    poly = np.asarray(polyline, dtype=np.float64)[:, :2]
    a, ab = poly[:-1], np.diff(poly, axis=0)
    length = np.maximum((ab**2).sum(1), 1e-18)
    best = np.full(len(pts), np.inf)
    for i in range(0, len(a), 1500):
        A, AB, L = a[i:i+1500], ab[i:i+1500], length[i:i+1500]
        t = np.clip(((pts[:, None, :]-A[None])*AB[None]).sum(-1)/L[None], 0, 1)
        d = np.sqrt(((pts[:, None, :]-(A[None]+t[..., None]*AB[None]))**2).sum(-1))
        best = np.minimum(best, d.min(1))
    return float(best.max())


def warm_start_features(path, cfg=DEFAULT, wheelbase_m=None):
    """Everything the Flow model needs from one warm start.

    ``k4_tokens`` keeps the historical key name used by the encoder and the
    'tokens' source; ``warm_latent``/``warm_structure`` feed the 'controls'
    source.  Returns (features, controls, info).
    """
    from .conditions import path_tokens
    from .numerical import encode, structure_id
    controls, info = controls_from_hybrid_astar(path, wheelbase_m, cfg)
    labelled, info['relabelled_gear_rows'] = canonical_gear_labels(path)
    latent, mask = encode(controls, cfg)
    features = dict(k4_tokens=path_tokens(labelled, cfg), warm_latent=latent.astype(np.float32),
                    warm_latent_mask=mask, warm_structure=np.int64(structure_id(controls.gears, cfg)))
    if not all(np.isfinite(v).all() for v in (features['k4_tokens'], features['warm_latent'])):
        raise ValueError('nonfinite_warm_start_features')
    return features, controls, info


def planner_representation(meta, cfg=DEFAULT):
    """Hybrid A* parameters recorded in a warm-start checkpoint representation."""
    pc, vehicle = meta['planner_config'], meta['vehicle']
    return dict(planner=meta['planner'], primitive_length_m=pc['primitive_length'],
                step_m=pc['integration_step'], steering_samples=pc['steering_samples'],
                max_expansions=pc['max_expansions'], wheelbase_m=vehicle['wheel_base'],
                max_steer_rad=vehicle['max_steer'], steering_reading=STEERING_READING,
                curvature_conversion=bool(abs(vehicle['wheel_base']-cfg.wheelbase) > 1e-9))


def warm_start_representation(base, meta, *, source_kind, gear_mismatch_policy,
                              residual_noise_scale=0.15, warm_start_source='hybrid_astar', cfg=DEFAULT):
    """Spline representation for 'warmstart_residual_v1' checkpoints."""
    return dict(base, flow_source='warmstart_residual_v1', source_kind=source_kind,
                warm_start_source=warm_start_source, residual_noise_scale=float(residual_noise_scale),
                gear_mismatch_policy=gear_mismatch_policy,
                warm_start_planner=planner_representation(meta, cfg),
                source_contract=('warm start converted to retimed controls on its own gear structure; '
                                 'other structures use the numeric-token source'
                                 if source_kind == 'controls' else 'numeric warm-start tokens only'))
