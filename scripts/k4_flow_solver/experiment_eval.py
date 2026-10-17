"""Common evaluator for experiments 1-4 (J, feasibility, timing, evaluation uids).

  1 Hybrid A* + Flow solver     2 Hybrid A* + IPOPT
  3 OCP GT (baseline)           4 Hybrid A* alone

Every method is scored by ``numerical.validate`` with one goal tolerance
(default 0.1 m / 5 deg): J = T + 0.5*integral(a^2 + v^2 steering_rate^2) dt,
feasibility = continuous collision/clearance, control limits, goal pose and
independent-integration agreement.  ``evaluate_path`` converts a Hybrid A*
path with ``warmstart.controls_from_hybrid_astar`` (split_path phases,
geometric interval curvature, retimed speed) - the same conversion as the
Flow x0 - and is used for experiment 4.

Evaluation uids are drawn from the validation split with a fixed seed.  They
exclude scenes used for design decisions and checkpoint selection but are
NOT an independent test set (the validation split may overlap earlier
64-scene evaluation panels of the K4 project).
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json
import math
import numpy as np

from .config import DEFAULT, SolverConfig
from .inference import goal_from_scene
from .numerical import validate

GOAL_POSITION_M = 0.1
GOAL_YAW_DEG = 5.0


def evaluation_config(cfg=None, *, position_m=GOAL_POSITION_M, yaw_deg=GOAL_YAW_DEG):
    return replace(cfg or DEFAULT, terminal_position_tolerance_m=float(position_m),
                   terminal_yaw_tolerance_rad=math.radians(float(yaw_deg)))


def evaluate_controls(controls, scene, goal=None, cfg=None, *, position_m=GOAL_POSITION_M, yaw_deg=GOAL_YAW_DEG):
    """J and verified feasibility of numerical ``Controls`` in the start-local frame."""
    cfg = evaluation_config(cfg, position_m=position_m, yaw_deg=yaw_deg)
    goal = goal_from_scene(scene) if goal is None else np.asarray(goal, dtype=float)
    try:
        check = validate(controls, goal, scene, cfg)
    except ValueError as exc:
        return dict(J=None, verified_feasible=False, reason=f'invalid_controls:{exc}',
                    goal_tolerance=dict(position_m=position_m, yaw_deg=yaw_deg))
    geometry = check['geometry'] or {}
    return dict(J=float(check['J']), verified_feasible=bool(check['verified_feasible']),
                terminal_ok=bool(check['terminal_ok']), controls_ok=bool(check['controls_ok']),
                collision_free=geometry.get('collision_free'), clearance_m=geometry.get('conservative_continuous_clearance_m'),
                endpoint_error_m=float(np.hypot(*check['endpoint_error'][:2])),
                endpoint_yaw_error_deg=float(abs(math.degrees(check['endpoint_error'][2]))),
                duration_s=float(check['duration_s']), gears=controls.gears.tolist(),
                goal_tolerance=dict(position_m=position_m, yaw_deg=yaw_deg))


def evaluate_path(path, scene, goal=None, cfg=None, *, position_m=GOAL_POSITION_M, yaw_deg=GOAL_YAW_DEG,
                  wheelbase_m=None):
    """Experiment 4: a Hybrid A* path (cache convention) -> retimed controls -> evaluate."""
    from .warmstart import controls_from_hybrid_astar
    base = cfg or DEFAULT
    try:
        controls, info = controls_from_hybrid_astar(np.asarray(path, dtype=float), wheelbase_m, base)
    except ValueError as exc:
        return dict(J=None, verified_feasible=False, reason=f'conversion_failed:{exc}',
                    goal_tolerance=dict(position_m=position_m, yaw_deg=yaw_deg))
    result = evaluate_controls(controls, scene, goal, base, position_m=position_m, yaw_deg=yaw_deg)
    result['conversion'] = info
    return result


def timing_record(*, warm_start_s, optimizer_s, device, warmup_excluded):
    """One timing format for all experiments (seconds; wall clock, CUDA-synchronized where relevant)."""
    warm = None if warm_start_s is None else float(warm_start_s)
    opt = None if optimizer_s is None else float(optimizer_s)
    total = None if warm is None and opt is None else (warm or 0.0)+(opt or 0.0)
    return dict(warm_start_s=warm, optimizer_s=opt, total_s=total, device=str(device),
                warmup_excluded=bool(warmup_excluded))


def make_eval_uids(dataset, count, seed, *, exclude=(), note=''):
    """Fixed-seed sample of validation uids (all validation scenes are eligible
    regardless of any method's success, so failures count for every method)."""
    rows = [json.loads(line) for line in (Path(dataset)/'sample_reports.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    excluded = set(exclude)
    pool = sorted(r['uid'] for r in rows if r.get('success') and r['split'] == 'validation' and r['uid'] not in excluded)
    if count > len(pool):
        raise ValueError('not enough validation scenes after exclusions')
    chosen = sorted(np.random.default_rng(seed).choice(pool, size=count, replace=False).tolist())
    return dict(schema='experiment-eval-uids-v1', split='validation', count=count, seed=seed,
                eligible=len(pool), excluded=len(excluded & {r['uid'] for r in rows}), uids=chosen,
                goal_tolerance=dict(position_m=GOAL_POSITION_M, yaw_deg=GOAL_YAW_DEG),
                independence=('Validation split, disjoint from the excluded design/selection scenes; '
                              'not an independent test set (may overlap earlier 64-scene panels)'),
                note=note)
