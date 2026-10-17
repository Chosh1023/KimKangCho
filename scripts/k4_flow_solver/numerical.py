"""Float64 reference controls, bounded repair, and teacher-compatible validation.

No optimizer is called by validation. Repairs change controls, never endpoint XY.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import time
import numpy as np
from scipy.integrate import solve_ivp

from .config import DEFAULT, SolverConfig


def grid(n):
    return 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, n + 1)))


@lru_cache(maxsize=8)
def quadrature(n):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return (nodes+1)/2, weights/2


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


@dataclass
class Controls:
    lengths: np.ndarray
    delta: np.ndarray
    b: np.ndarray
    gears: np.ndarray

    def copy(self):
        return Controls(*(v.copy() for v in (self.lengths, self.delta, self.b, self.gears)))

    def check(self, cfg=DEFAULT):
        m = len(self.lengths)
        if not 1 <= m <= cfg.max_phases:
            raise ValueError("phase_count_out_of_scope")
        if self.delta.shape != self.b.shape or self.delta.shape[0] != m or self.delta.shape[1] < 3:
            raise ValueError("invalid_control_shapes")
        if self.gears.shape != (m,) or not np.isin(self.gears, [-1, 1]).all():
            raise ValueError("invalid_gears")
        if not np.all(self.gears[1:] == -self.gears[:-1]):
            raise ValueError("non_alternating_phases")
        if not all(np.isfinite(a).all() for a in (self.lengths, self.delta, self.b)):
            raise ValueError("nonfinite_controls")
        if np.any(self.lengths <= 1e-6) or self.lengths.sum() > cfg.max_total_length:
            raise ValueError("length_out_of_scope")
        if np.any(self.b < 0) or np.any(self.b[:, [0, -1]] != 0):
            raise ValueError("invalid_speed_boundary")
        if np.any(self.b[:, :-1] + self.b[:, 1:] <= 0):
            raise ValueError("stationary_positive_distance_interval")
        if np.max(np.abs(self.delta)) >= np.pi / 2:
            raise ValueError("singular_steering")


def structure_id(gears, cfg=DEFAULT):
    gears = np.asarray(gears)
    if not 1 <= len(gears) <= cfg.max_phases or not np.isin(gears, [-1, 1]).all():
        raise ValueError("invalid_structure")
    if not np.all(gears[1:] == -gears[:-1]):
        raise ValueError("non_alternating_phases")
    return (int(gears[0] < 0) * cfg.max_phases) + len(gears) - 1


def gears_for(h, cfg=DEFAULT):
    if not 0 <= int(h) < 2 * cfg.max_phases:
        raise ValueError("invalid_structure_id")
    m = int(h) % cfg.max_phases + 1
    return (1 if int(h) < cfg.max_phases else -1) * (-1) ** np.arange(m)


def encode(c, cfg=DEFAULT):
    c.check(cfg)
    m = len(c.lengths)
    e = cfg.latent_epsilon
    latent = np.zeros((cfg.max_phases, cfg.intervals + 1, 3), dtype=np.float64)
    mask = np.zeros_like(latent, dtype=bool)
    if c.delta.shape[1] != cfg.intervals + 1:
        raise ValueError("grid_mismatch")
    # Length has one degree of freedom per phase; no repeated-length loss.
    latent[:m, 0, 0] = np.log(c.lengths)
    latent[:m, :, 1] = np.arctanh(np.clip(c.delta / cfg.max_steer, -1 + e, 1 - e))
    f = np.clip(c.b[:, 1:-1] / cfg.max_speed**2, e, 1 - e)
    latent[:m, 1:-1, 2] = np.log(f) - np.log1p(-f)
    mask[:m, 0, 0] = True
    mask[:m, :, 1] = True
    mask[:m, 1:-1, 2] = True
    return latent, mask


def decode(latent, h, cfg=DEFAULT):
    from scipy.special import expit
    a = np.asarray(latent, dtype=np.float64)
    if a.shape != (cfg.max_phases, cfg.intervals + 1, 3) or not np.isfinite(a).all():
        raise ValueError("invalid_latent")
    gears = gears_for(h, cfg)
    m = len(gears)
    if np.any(a[:m, 0, 0] > np.log(cfg.max_total_length)):
        raise ValueError("length_overflow")
    lengths = np.exp(a[:m, 0, 0])
    delta = cfg.max_steer * np.tanh(a[:m, :, 1])
    b = cfg.max_speed**2 * expit(a[:m, :, 2])
    b[:, [0, -1]] = 0
    c = Controls(lengths, delta, b, gears)
    c.check(cfg)
    return c


def integrate(c, *, dense_spacing=None, cfg=DEFAULT):
    """Analytic yaw + Gaussian XY integration, keeping every control knot."""
    c.check(cfg)
    state = np.zeros(3, dtype=np.float64)
    phases, arcs = [], []
    for length, d, gear in zip(c.lengths, c.delta, c.gears):
        u = grid(len(d) - 1)
        if dense_spacing is not None:
            counts = np.maximum(1, np.ceil(length * np.diff(u) / dense_spacing).astype(int))
            new_u = np.r_[np.concatenate([np.linspace(a, b, n, endpoint=False)
                                         for a, b, n in zip(u[:-1], u[1:], counts)]), 1.0]
            d = np.interp(new_u, u, d)
            u = new_u
        h = length * np.diff(u)
        dd, d0 = np.diff(d), d[:-1]

        def turn(t):
            t = np.atleast_1d(t)[None, :]
            # log1p avoids cancellation for almost constant steering.
            varying = np.abs(dd) > 1e-9
            safe = np.where(varying, dd, 1.0)[:, None]
            dx = dd[:, None] * t
            logarithm = np.log1p(-2 * np.sin(dx / 2)**2 - np.tan(d0[:, None]) * np.sin(dx))
            varying_turn = -h[:, None] / safe * logarithm
            constant_turn = h[:, None] * t * np.tan(d0[:, None] + 0.5 * dd[:, None] * t)
            return gear / cfg.wheelbase * np.where(varying[:, None], varying_turn, constant_turn)

        theta = state[2] + np.r_[0.0, np.cumsum(turn([1.0])[:, 0])]
        nodes, weights = quadrature(8 if dense_spacing else 16)
        angle = theta[:-1, None] + turn(nodes)
        x = state[0] + np.r_[0.0, np.cumsum(gear * h * (np.cos(angle) @ weights))]
        y = state[1] + np.r_[0.0, np.cumsum(gear * h * (np.sin(angle) @ weights))]
        phases.append(np.c_[x, y, theta, np.full(len(x), gear), d])
        arcs.append(length * u)
        state = phases[-1][-1, :3].copy()
    return state, phases, arcs


def cost(c, cfg=DEFAULT, start_delta=0.0, end_delta=0.0):
    c.check(cfg)
    h = c.lengths[:, None] * np.diff(grid(c.b.shape[1] - 1))[None, :]
    q = np.sqrt(c.b)
    q0, q1 = q[:, :-1], q[:, 1:]
    dt = 2 * h / (q0 + q1)
    a = np.diff(c.b, axis=1) / (2 * h)
    slope = np.diff(c.delta, axis=1) / h
    f4 = (q0**4 + q0**3*q1 + q0*q0*q1*q1 + q0*q1**3 + q1**4) / 5
    ac, sc = float(np.sum(dt*a*a)), float(np.sum(dt*slope*slope*f4))
    changes = np.r_[c.delta[0, 0] - start_delta,
                    c.delta[1:, 0] - c.delta[:-1, -1], end_delta - c.delta[-1, -1]]
    dwell = np.abs(changes) / (cfg.max_steer_rate - cfg.control_margin)
    duration = float(dt.sum() + dwell.sum())
    maxima = dict(max_speed=float(q.max()), max_accel=float(np.abs(a).max()),
                  max_steer=float(np.abs(c.delta).max()),
                  max_steer_rate=float((np.abs(slope)*np.maximum(q0, q1)).max()))
    return dict(J=duration + 0.5*(ac+sc), duration_s=duration, acceleration_integral=ac,
                steering_integral=sc, dwell_s=dwell.tolist(), **maxima,
                controls_ok=bool(maxima['max_speed'] <= cfg.max_speed + 1e-8 and
                    maxima['max_accel'] <= cfg.max_accel + 1e-8 and
                    maxima['max_steer'] <= cfg.max_steer + 1e-8 and
                    maxima['max_steer_rate'] <= cfg.max_steer_rate + 1e-8))


def retime(c, cfg=DEFAULT):
    """Solver-free forward/backward caps, preserving the proposed speed shape."""
    out = c.copy()
    h = out.lengths[:, None] * np.diff(grid(out.b.shape[1]-1))
    slope = np.abs(np.diff(out.delta, axis=1)) / h
    cap = ((cfg.max_steer_rate-cfg.control_margin) / np.maximum(slope, 1e-12))**2
    b = np.minimum(out.b, (cfg.max_speed-cfg.control_margin)**2)
    b[:, :-1] = np.minimum(b[:, :-1], cap)
    b[:, 1:] = np.minimum(b[:, 1:], cap)
    b[:, [0, -1]] = 0
    accel = cfg.max_accel-cfg.control_margin
    for i in range(1, b.shape[1]):
        b[:, i] = np.minimum(b[:, i], b[:, i-1]+2*accel*h[:, i-1])
    for i in range(b.shape[1]-2, -1, -1):
        b[:, i] = np.minimum(b[:, i], b[:, i+1]+2*accel*h[:, i])
    out.b = b
    out.check(cfg)
    return out


def terminal_satisfied(error, cfg=DEFAULT):
    return bool(np.linalg.norm(error[:2]) < cfg.terminal_position_tolerance_m and
                abs(error[2]) < cfg.terminal_yaw_tolerance_rad)


def repair(c, goal, cfg=DEFAULT, *, max_iterations=5, length_bound=0.05,
           steering_bound=0.01, time_budget_s=0.15):
    """Minimum-norm damped correction in length/constant/sine steering bases."""
    c.check(cfg)
    goal = np.asarray(goal)
    m = len(c.lengths)
    basis = np.sin(np.pi*grid(c.delta.shape[1]-1))
    x = np.zeros(3*m)
    started = time.perf_counter()

    def candidate(values):
        out = c.copy()
        out.lengths += values[:m]
        out.delta += values[m:2*m, None] + values[2*m:, None]*basis
        return out

    def error(values):
        end = integrate(candidate(values), cfg=cfg)[0]
        return np.r_[end[:2]-goal[:2], wrap(end[2]-goal[2])]

    iterations = 0
    for iterations in range(max_iterations):
        e = error(x)
        # Practical parking tolerances stop shooting once the requested pose
        # region is reached. Keep the historic strict stopping rule otherwise.
        practical = cfg.terminal_position_tolerance_m>1e-7 or cfg.terminal_yaw_tolerance_rad>1e-7
        if (terminal_satisfied(e,cfg) if practical else np.linalg.norm(e)<1e-9) or time.perf_counter()-started >= time_budget_s:
            break
        step = 1e-5
        jac = np.column_stack([(error(x+step*np.eye(3*m)[j])-e)/step for j in range(3*m)])
        # Nearly symmetric/straight phases have redundant steering bases.
        # Do not amplify finite-difference noise along their null directions.
        direction = np.linalg.lstsq(jac, -e, rcond=1e-6)[0]
        accepted = False
        for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial = x+fraction*direction
            out = candidate(trial)
            if (np.max(np.abs(trial[:m])) > length_bound or
                np.max(np.abs(out.delta-c.delta)) > steering_bound or
                np.any(out.lengths <= 1e-6) or
                np.max(np.abs(out.delta)) > cfg.max_steer):
                continue
            if np.linalg.norm(error(trial)) < np.linalg.norm(e):
                x = trial
                accepted = True
                break
        if not accepted:
            break
    out = retime(candidate(x), cfg)
    e = error(x)
    return out, dict(iterations=iterations+1, seconds=time.perf_counter()-started,
                     endpoint_error=e.tolist(), endpoint_ok=terminal_satisfied(e,cfg),
                     max_length_correction=float(np.abs(x[:m]).max()),
                     max_steer_correction=float(np.abs(out.delta-c.delta).max()))


def validate(c, goal, scene, cfg=DEFAULT, *, start_delta=0.0, end_delta=0.0):
    """Configured goal tolerance; unchanged dense geometry and ODE agreement checks."""
    started = time.perf_counter()
    c.check(cfg)
    metrics = cost(c, cfg, start_delta, end_delta)
    end, _, _ = integrate(c, cfg=cfg)
    error = np.r_[end[:2]-goal[:2], wrap(end[2]-goal[2])]
    terminal = terminal_satisfied(error,cfg)
    result = dict(**metrics, endpoint_error=error.tolist(), terminal_ok=bool(terminal),
                  geometry=None, independent_endpoint_error=None, independent_integration_difference=None,
                  verified_feasible=False,goal_tolerance=dict(position_m=cfg.terminal_position_tolerance_m,
                      yaw_rad=cfg.terminal_yaw_tolerance_rad))
    # Reject cheap failures before the expensive dense footprint/independent ODE.
    if terminal and metrics['controls_ok']:
        from certify_k4_small_scene_cost_bounds import check_geometry
        _, phases, _ = integrate(c, dense_spacing=0.001, cfg=cfg)
        path = np.vstack(phases)
        geometry = check_geometry(path, scene, float(np.abs(np.tan(c.delta)).max())/cfg.wheelbase)
        result['geometry'] = geometry
        if geometry['collision_free'] and geometry['final_inside'] and geometry['conservative_continuous_clearance_m'] >= cfg.clearance:
            state = np.zeros(3)
            ok = True
            state, ok = independent_endpoint(c, cfg)
            independent = np.r_[state[:2]-goal[:2], wrap(state[2]-goal[2])]
            result['independent_endpoint_error'] = independent.tolist()
            difference = np.r_[state[:2]-end[:2],wrap(state[2]-end[2])]
            result['independent_integration_difference'] = difference.tolist()
            # Numerical integrators must still agree at micrometer precision.
            # Their common endpoint may legitimately be within 10 cm of goal.
            result['verified_feasible'] = bool(ok and np.linalg.norm(difference[:2]) < 3e-6 and abs(difference[2]) < 1e-6
                and np.linalg.norm(independent[:2]) < max(cfg.terminal_position_tolerance_m,3e-6)
                and abs(independent[2]) < max(cfg.terminal_yaw_tolerance_rad,1e-6))
    result['validation_seconds'] = time.perf_counter()-started
    return result


def independent_endpoint(c, cfg=DEFAULT):
    """DOP853 solves each PL interval, batched on u in [0,1].

    A single adaptive solve in physical s can step over narrow cosine-grid
    steering corners. Giving every interval its own state prevents that alias
    without loosening tolerances or reusing the analytic yaw integrator.
    """
    state = np.zeros(3)
    ok = True
    for length, d, gear in zip(c.lengths, c.delta, c.gears):
        h = length*np.diff(grid(len(d)-1))
        n = len(h)
        def rhs(u, flat):
            a = flat.reshape(n, 3)
            return np.c_[gear*h*np.cos(a[:, 2]), gear*h*np.sin(a[:, 2]),
                gear*h*np.tan(d[:-1]+u*np.diff(d))/cfg.wheelbase].ravel()
        solved = solve_ivp(rhs, [0., 1.], np.zeros(3*n), method='DOP853',
                           rtol=2e-10, atol=1e-12, max_step=0.25)
        increments = solved.y[:, -1].reshape(n, 3)
        heading = state[2]+np.r_[0., np.cumsum(increments[:-1, 2])]
        state[:2] += [np.sum(np.cos(heading)*increments[:, 0]-np.sin(heading)*increments[:, 1]),
                      np.sum(np.sin(heading)*increments[:, 0]+np.cos(heading)*increments[:, 1])]
        state[2] += increments[:, 2].sum()
        ok = ok and solved.success
    return state, ok


def from_teacher(path, cfg=DEFAULT):
    with np.load(path, allow_pickle=False) as a:
        lengths, gears = a['lengths'].astype(float), a['gears'].astype(int)
        offset = a['phase_control_offsets']
        delta, b = [], []
        for lo, hi in zip(offset[:-1], offset[1:]):
            old = grid(int(hi-lo)-1)
            delta.append(np.interp(grid(cfg.intervals), old, a['delta'][lo:hi]))
            b.append(np.interp(grid(cfg.intervals), old, a['speed'][lo:hi]**2))
        return Controls(lengths, np.asarray(delta), np.asarray(b), gears), a['goal_local'].astype(float)


def from_k4(path, cfg=DEFAULT):
    from fast_multiphase_ocp import split_path
    parts, gears = split_path(path)
    if len(parts) > cfg.max_phases:
        raise ValueError('k4_baseline_phase_count_out_of_scope')
    lengths = np.asarray([s[-1] for _, s in parts])
    delta = np.asarray([np.interp(grid(cfg.intervals)*s[-1], s, p[:, 4]) for p, s in parts])
    b = np.full_like(delta, (cfg.max_speed-cfg.control_margin)**2)
    b[:, [0, -1]] = 0
    return retime(Controls(lengths, delta, b, np.asarray(gears)), cfg)


def time_trace(c, cfg=DEFAULT, *, start_delta=0.0, end_delta=0.0, spacing=0.02):
    """Return [time,x,y,yaw,signed_speed,steering,gear], including dwell."""
    _, phases, arcs = integrate(c, dense_spacing=spacing, cfg=cfg)
    result, clock, previous = [], 0.0, start_delta
    for p, arc, length, delta, b, gear in zip(phases, arcs, c.lengths, c.delta, c.b, c.gears):
        pose = p[0, :3]
        duration = abs(delta[0]-previous)/(cfg.max_steer_rate-cfg.control_margin)
        result.append([clock, *pose, 0.0, previous, gear])
        if duration > 0:
            clock += duration
            result.append([clock, *pose, 0.0, delta[0], gear])
        knots = length*grid(len(b)-1)
        q = np.sqrt(b)
        dt = 2*np.diff(knots)/(q[:-1]+q[1:])
        tknots = np.r_[0.0, np.cumsum(dt)]
        indices = np.clip(np.searchsorted(knots, arc, side='right')-1, 0, len(b)-2)
        local_b = np.interp(arc, knots, b)
        ds = arc-knots[indices]
        denom = q[indices]+np.sqrt(local_b)
        elapsed = np.divide(2*ds, denom, out=np.zeros_like(ds), where=denom>0)
        times = clock+tknots[indices]+elapsed
        result.extend(np.c_[times[1:], p[1:, :3], gear*np.sqrt(local_b[1:]), p[1:, 4], np.full(len(p)-1, gear)].tolist())
        clock += tknots[-1]
        previous = delta[-1]
    duration = abs(end_delta-previous)/(cfg.max_steer_rate-cfg.control_margin)
    if duration > 0:
        clock += duration
        result.append([clock, *phases[-1][-1, :3], 0.0, end_delta, c.gears[-1]])
    return np.asarray(result, dtype=np.float64)
