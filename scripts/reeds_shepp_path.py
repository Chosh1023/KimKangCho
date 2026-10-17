#!/usr/bin/env python3
"""Dependency-free Reeds-Shepp path generation.

The implementation enumerates the complete canonical Reeds-Shepp word set
(SCS, CSC, CCC, CCCC, CCSC and CCSCC) and samples every candidate with exact
constant-curvature integration.  Segment lengths are returned in metres;
positive lengths are forward and negative lengths are reverse.

The formulas are adapted from the Reeds-Shepp implementation already shipped
in ``CAAPolicy/agent/hybrid_A_star.pyx``.  This module removes the Cython,
SciPy and heapdict dependencies and fixes path de-duplication and metric cost
handling so it can be used by offline BEV tools and the CARLA runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple


LEFT = "L"
RIGHT = "R"
STRAIGHT = "S"


@dataclass
class ReedsSheppPath:
    lengths: List[float]
    ctypes: List[str]
    total_length: float = 0.0
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    yaw: List[float] = field(default_factory=list)
    directions: List[int] = field(default_factory=list)
    steers: List[float] = field(default_factory=list)

    @property
    def gear_switches(self) -> int:
        signs = [1 if value >= 0.0 else -1 for value in self.lengths if abs(value) > 1e-9]
        return sum(a != b for a, b in zip(signs, signs[1:]))


def wrap_pi(theta: float) -> float:
    return (float(theta) + math.pi) % (2.0 * math.pi) - math.pi


def _polar(x: float, y: float) -> Tuple[float, float]:
    return math.hypot(x, y), math.atan2(y, x)


def _mod2pi(theta: float) -> float:
    return wrap_pi(theta)


def _set_path(
    paths: List[ReedsSheppPath],
    lengths: Sequence[float],
    ctypes: Sequence[str],
    max_normalized_length: float,
) -> List[ReedsSheppPath]:
    lengths = [float(value) for value in lengths]
    ctypes = [str(value) for value in ctypes]
    total = sum(abs(value) for value in lengths)
    if total < 1e-6 or total > float(max_normalized_length):
        return paths
    for existing in paths:
        if existing.ctypes == ctypes and len(existing.lengths) == len(lengths):
            if sum(abs(a - b) for a, b in zip(existing.lengths, lengths)) <= 1e-5:
                return paths
    paths.append(ReedsSheppPath(lengths=lengths, ctypes=ctypes, total_length=total))
    return paths


def _lsl(x: float, y: float, phi: float) -> Tuple[bool, float, float, float]:
    u, t = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if t >= 0.0:
        v = _mod2pi(phi - t)
        if v >= 0.0:
            return True, t, u, v
    return False, 0.0, 0.0, 0.0


def _lsr(x: float, y: float, phi: float) -> Tuple[bool, float, float, float]:
    u1, t1 = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    squared = u1 * u1
    if squared >= 4.0:
        u = math.sqrt(max(0.0, squared - 4.0))
        theta = math.atan2(2.0, u)
        t = _mod2pi(t1 + theta)
        v = _mod2pi(t - phi)
        if t >= 0.0 and v >= 0.0:
            return True, t, u, v
    return False, 0.0, 0.0, 0.0


def _lrl(x: float, y: float, phi: float) -> Tuple[bool, float, float, float]:
    u1, t1 = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if u1 <= 4.0:
        u = -2.0 * math.asin(max(-1.0, min(1.0, 0.25 * u1)))
        t = _mod2pi(t1 + 0.5 * u + math.pi)
        v = _mod2pi(phi - t + u)
        if t >= 0.0 and u <= 0.0:
            return True, t, u, v
    return False, 0.0, 0.0, 0.0


def _sls(x: float, y: float, phi: float) -> Tuple[bool, float, float, float]:
    phi = _mod2pi(phi)
    if abs(math.tan(phi)) < 1e-10:
        return False, 0.0, 0.0, 0.0
    if y > 0.0 and 0.0 < phi < math.pi * 0.99:
        xd = -y / math.tan(phi) + x
        t = xd - math.tan(phi / 2.0)
        u = phi
        v = math.hypot(x - xd, y) - math.tan(phi / 2.0)
        return True, t, u, v
    if y < 0.0 and 0.0 < phi < math.pi * 0.99:
        xd = -y / math.tan(phi) + x
        t = xd - math.tan(phi / 2.0)
        u = phi
        v = -math.hypot(x - xd, y) - math.tan(phi / 2.0)
        return True, t, u, v
    return False, 0.0, 0.0, 0.0


def _scs(x: float, y: float, phi: float, paths: List[ReedsSheppPath], limit: float) -> None:
    flag, t, u, v = _sls(x, y, phi)
    if flag:
        _set_path(paths, [t, u, v], [STRAIGHT, LEFT, STRAIGHT], limit)
    flag, t, u, v = _sls(x, -y, -phi)
    if flag:
        _set_path(paths, [t, u, v], [STRAIGHT, RIGHT, STRAIGHT], limit)


def _csc(x: float, y: float, phi: float, paths: List[ReedsSheppPath], limit: float) -> None:
    variants = (
        (_lsl(x, y, phi), (1, 1, 1), (LEFT, STRAIGHT, LEFT)),
        (_lsl(-x, y, -phi), (-1, -1, -1), (LEFT, STRAIGHT, LEFT)),
        (_lsl(x, -y, -phi), (1, 1, 1), (RIGHT, STRAIGHT, RIGHT)),
        (_lsl(-x, -y, phi), (-1, -1, -1), (RIGHT, STRAIGHT, RIGHT)),
        (_lsr(x, y, phi), (1, 1, 1), (LEFT, STRAIGHT, RIGHT)),
        (_lsr(-x, y, -phi), (-1, -1, -1), (LEFT, STRAIGHT, RIGHT)),
        (_lsr(x, -y, -phi), (1, 1, 1), (RIGHT, STRAIGHT, LEFT)),
        (_lsr(-x, -y, phi), (-1, -1, -1), (RIGHT, STRAIGHT, LEFT)),
    )
    for result, signs, ctypes in variants:
        flag, t, u, v = result
        if flag:
            _set_path(paths, [t * signs[0], u * signs[1], v * signs[2]], ctypes, limit)


def _ccc(x: float, y: float, phi: float, paths: List[ReedsSheppPath], limit: float) -> None:
    variants = (
        (_lrl(x, y, phi), (1, 1, 1), (LEFT, RIGHT, LEFT), False),
        (_lrl(-x, y, -phi), (-1, -1, -1), (LEFT, RIGHT, LEFT), False),
        (_lrl(x, -y, -phi), (1, 1, 1), (RIGHT, LEFT, RIGHT), False),
        (_lrl(-x, -y, phi), (-1, -1, -1), (RIGHT, LEFT, RIGHT), False),
    )
    for result, signs, ctypes, _ in variants:
        flag, t, u, v = result
        if flag:
            _set_path(paths, [t * signs[0], u * signs[1], v * signs[2]], ctypes, limit)

    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)
    backward = (
        (_lrl(xb, yb, phi), (1, 1, 1), (LEFT, RIGHT, LEFT)),
        (_lrl(-xb, yb, -phi), (-1, -1, -1), (LEFT, RIGHT, LEFT)),
        (_lrl(xb, -yb, -phi), (1, 1, 1), (RIGHT, LEFT, RIGHT)),
        (_lrl(-xb, -yb, phi), (-1, -1, -1), (RIGHT, LEFT, RIGHT)),
    )
    for result, signs, ctypes in backward:
        flag, t, u, v = result
        if flag:
            _set_path(paths, [v * signs[0], u * signs[1], t * signs[2]], ctypes, limit)


def _tau_omega(u: float, v: float, xi: float, eta: float, phi: float) -> Tuple[float, float]:
    delta = _mod2pi(u - v)
    a = math.sin(u) - math.sin(delta)
    b = math.cos(u) - math.cos(delta) - 1.0
    t1 = math.atan2(eta * a - xi * b, xi * a + eta * b)
    t2 = 2.0 * (math.cos(delta) - math.cos(v) - math.cos(u)) + 3.0
    tau = _mod2pi(t1 + math.pi) if t2 < 0.0 else _mod2pi(t1)
    omega = _mod2pi(tau - u + v - phi)
    return tau, omega


def _lrlrn(x: float, y: float, phi: float) -> Tuple[bool, float, float, float]:
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho = 0.25 * (2.0 + math.hypot(xi, eta))
    if rho <= 1.0:
        u = math.acos(max(-1.0, min(1.0, rho)))
        t, v = _tau_omega(u, -u, xi, eta, phi)
        if t >= 0.0 and v <= 0.0:
            return True, t, u, v
    return False, 0.0, 0.0, 0.0


def _lrlrp(x: float, y: float, phi: float) -> Tuple[bool, float, float, float]:
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho = (20.0 - xi * xi - eta * eta) / 16.0
    if 0.0 <= rho <= 1.0:
        u = -math.acos(max(-1.0, min(1.0, rho)))
        if u >= -0.5 * math.pi:
            t, v = _tau_omega(u, u, xi, eta, phi)
            if t >= 0.0 and v >= 0.0:
                return True, t, u, v
    return False, 0.0, 0.0, 0.0


def _cccc(x: float, y: float, phi: float, paths: List[ReedsSheppPath], limit: float) -> None:
    # The sign symmetries of the two four-curve formula families are easy to
    # mistranscribe (the legacy Cython copy in this repository contains that
    # exact bug).  Enumerate the finite sign/type symmetries and retain only
    # combinations whose exact normalized endpoint is (x, y, phi).
    parameter_sets = []
    transforms = ((x, y, phi), (-x, y, -phi), (x, -y, -phi), (-x, -y, phi))
    for tx, ty, tphi in transforms:
        for solver in (_lrlrn, _lrlrp):
            flag, t, u, v = solver(tx, ty, tphi)
            if flag:
                parameter_sets.append((abs(t), abs(u), abs(v)))

    def endpoint(lengths: Sequence[float], ctypes: Sequence[str]) -> Tuple[float, float, float]:
        px = py = pyaw = 0.0
        for length, ctype in zip(lengths, ctypes):
            px, py, pyaw = _interpolate_local(px, py, pyaw, length, ctype, 1.0)
        return px, py, wrap_pi(pyaw)

    for t, u, v in parameter_sets:
        for ctypes in ((LEFT, RIGHT, LEFT, RIGHT), (RIGHT, LEFT, RIGHT, LEFT)):
            for mask in range(16):
                signs = [1.0 if mask & (1 << index) else -1.0 for index in range(4)]
                values = [signs[0] * t, signs[1] * u, signs[2] * u, signs[3] * v]
                ex, ey, eyaw = endpoint(values, ctypes)
                if math.hypot(ex - x, ey - y) <= 1e-6 and abs(wrap_pi(eyaw - phi)) <= 1e-6:
                    _set_path(paths, values, ctypes, limit)


def _lrsr(x: float, y: float, phi: float) -> Tuple[bool, float, float, float]:
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho, theta = _polar(-eta, xi)
    if rho >= 2.0:
        t = theta
        u = 2.0 - rho
        v = _mod2pi(t + 0.5 * math.pi - phi)
        if t >= 0.0 and u <= 0.0 and v <= 0.0:
            return True, t, u, v
    return False, 0.0, 0.0, 0.0


def _lrsl(x: float, y: float, phi: float) -> Tuple[bool, float, float, float]:
    xi = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    rho, theta = _polar(xi, eta)
    if rho >= 2.0:
        root = math.sqrt(max(0.0, rho * rho - 4.0))
        u = 2.0 - root
        t = _mod2pi(theta + math.atan2(root, -2.0))
        v = _mod2pi(phi - 0.5 * math.pi - t)
        if t >= 0.0 and u <= 0.0 and v <= 0.0:
            return True, t, u, v
    return False, 0.0, 0.0, 0.0


def _ccsc(x: float, y: float, phi: float, paths: List[ReedsSheppPath], limit: float) -> None:
    half = 0.5 * math.pi
    variants = (
        (_lrsl(x, y, phi), [1, -half, 1, 1], (LEFT, RIGHT, STRAIGHT, LEFT), False),
        (_lrsl(-x, y, -phi), [-1, half, -1, -1], (LEFT, RIGHT, STRAIGHT, LEFT), False),
        (_lrsl(x, -y, -phi), [1, -half, 1, 1], (RIGHT, LEFT, STRAIGHT, RIGHT), False),
        (_lrsl(-x, -y, phi), [-1, half, -1, -1], (RIGHT, LEFT, STRAIGHT, RIGHT), False),
        (_lrsr(x, y, phi), [1, -half, 1, 1], (LEFT, RIGHT, STRAIGHT, RIGHT), False),
        (_lrsr(-x, y, -phi), [-1, half, -1, -1], (LEFT, RIGHT, STRAIGHT, RIGHT), False),
        (_lrsr(x, -y, -phi), [1, -half, 1, 1], (RIGHT, LEFT, STRAIGHT, LEFT), False),
        (_lrsr(-x, -y, phi), [-1, half, -1, -1], (RIGHT, LEFT, STRAIGHT, LEFT), False),
    )
    for result, pattern, ctypes, _ in variants:
        flag, t, u, v = result
        if flag:
            values = [pattern[0] * t, pattern[1], pattern[2] * u, pattern[3] * v]
            _set_path(paths, values, ctypes, limit)

    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)
    backward = (
        (_lrsl(xb, yb, phi), (LEFT, STRAIGHT, RIGHT, LEFT), (1, 1, -half, 1)),
        (_lrsl(-xb, yb, -phi), (LEFT, STRAIGHT, RIGHT, LEFT), (-1, -1, half, -1)),
        (_lrsl(xb, -yb, -phi), (RIGHT, STRAIGHT, LEFT, RIGHT), (1, 1, -half, 1)),
        (_lrsl(-xb, -yb, phi), (RIGHT, STRAIGHT, LEFT, RIGHT), (-1, -1, half, -1)),
        (_lrsr(xb, yb, phi), (RIGHT, STRAIGHT, RIGHT, LEFT), (1, 1, -half, 1)),
        (_lrsr(-xb, yb, -phi), (RIGHT, STRAIGHT, RIGHT, LEFT), (-1, -1, half, -1)),
        (_lrsr(xb, -yb, -phi), (LEFT, STRAIGHT, LEFT, RIGHT), (1, 1, -half, 1)),
        (_lrsr(-xb, -yb, phi), (LEFT, STRAIGHT, LEFT, RIGHT), (-1, -1, half, -1)),
    )
    for result, ctypes, signs in backward:
        flag, t, u, v = result
        if flag:
            _set_path(paths, [signs[0] * v, signs[1] * u, signs[2], signs[3] * t], ctypes, limit)


def _lrslr(x: float, y: float, phi: float) -> Tuple[bool, float, float, float]:
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho, _ = _polar(xi, eta)
    if rho >= 2.0:
        u = 4.0 - math.sqrt(max(0.0, rho * rho - 4.0))
        if u <= 0.0:
            t = _mod2pi(math.atan2((4.0 - u) * xi - 2.0 * eta, -2.0 * xi + (u - 4.0) * eta))
            v = _mod2pi(t - phi)
            if t >= 0.0 and v >= 0.0:
                return True, t, u, v
    return False, 0.0, 0.0, 0.0


def _ccscc(x: float, y: float, phi: float, paths: List[ReedsSheppPath], limit: float) -> None:
    half = 0.5 * math.pi
    variants = (
        (_lrslr(x, y, phi), [1, -half, 1, -half, 1], (LEFT, RIGHT, STRAIGHT, LEFT, RIGHT)),
        (_lrslr(-x, y, -phi), [-1, half, -1, half, -1], (LEFT, RIGHT, STRAIGHT, LEFT, RIGHT)),
        (_lrslr(x, -y, -phi), [1, -half, 1, -half, 1], (RIGHT, LEFT, STRAIGHT, RIGHT, LEFT)),
        (_lrslr(-x, -y, phi), [-1, half, -1, half, -1], (RIGHT, LEFT, STRAIGHT, RIGHT, LEFT)),
    )
    for result, pattern, ctypes in variants:
        flag, t, u, v = result
        if flag:
            _set_path(paths, [pattern[0] * t, pattern[1], pattern[2] * u, pattern[3], pattern[4] * v], ctypes, limit)


def generate_paths(
    start: Sequence[float],
    goal: Sequence[float],
    max_curvature: float,
    max_length_m: float = 80.0,
) -> List[ReedsSheppPath]:
    """Enumerate unsampled Reeds-Shepp candidates between two poses."""

    if max_curvature <= 0.0:
        raise ValueError("max_curvature must be positive")
    sx, sy, syaw = map(float, start[:3])
    gx, gy, gyaw = map(float, goal[:3])
    dx = gx - sx
    dy = gy - sy
    c = math.cos(syaw)
    s = math.sin(syaw)
    x = (c * dx + s * dy) * max_curvature
    y = (-s * dx + c * dy) * max_curvature
    phi = wrap_pi(gyaw - syaw)
    normalized_limit = max(1e-6, float(max_length_m) * max_curvature)
    paths: List[ReedsSheppPath] = []
    _scs(x, y, phi, paths, normalized_limit)
    _csc(x, y, phi, paths, normalized_limit)
    _ccc(x, y, phi, paths, normalized_limit)
    _cccc(x, y, phi, paths, normalized_limit)
    _ccsc(x, y, phi, paths, normalized_limit)
    _ccscc(x, y, phi, paths, normalized_limit)
    for path in paths:
        path.lengths = [value / max_curvature for value in path.lengths]
        path.total_length = sum(abs(value) for value in path.lengths)
    return paths


def _interpolate_local(
    ox: float,
    oy: float,
    oyaw: float,
    travel_normalized: float,
    ctype: str,
    max_curvature: float,
) -> Tuple[float, float, float]:
    value = float(travel_normalized)
    if ctype == STRAIGHT:
        return (
            ox + value / max_curvature * math.cos(oyaw),
            oy + value / max_curvature * math.sin(oyaw),
            oyaw,
        )
    ldx = math.sin(value) / max_curvature
    if ctype == LEFT:
        ldy = (1.0 - math.cos(value)) / max_curvature
        yaw = oyaw + value
    elif ctype == RIGHT:
        ldy = -(1.0 - math.cos(value)) / max_curvature
        yaw = oyaw - value
    else:
        raise ValueError("unknown Reeds-Shepp segment type: {}".format(ctype))
    c = math.cos(oyaw)
    s = math.sin(oyaw)
    return ox + c * ldx - s * ldy, oy + s * ldx + c * ldy, yaw


def sample_path(
    path: ReedsSheppPath,
    start: Sequence[float],
    max_curvature: float,
    step_size: float = 0.10,
    max_steer: float = 0.0,
) -> ReedsSheppPath:
    """Sample a candidate in global coordinates using exact arc equations."""

    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    sx, sy, syaw = map(float, start[:3])
    local_x = [0.0]
    local_y = [0.0]
    local_yaw = [0.0]
    directions = [1 if path.lengths[0] >= 0.0 else -1]
    first_steer = max_steer if path.ctypes[0] == LEFT else -max_steer if path.ctypes[0] == RIGHT else 0.0
    steers = [first_steer]

    for length_m, ctype in zip(path.lengths, path.ctypes):
        direction = 1 if length_m >= 0.0 else -1
        segment_m = abs(float(length_m))
        steps = max(1, int(math.ceil(segment_m / step_size)))
        ox, oy, oyaw = local_x[-1], local_y[-1], local_yaw[-1]
        steer = max_steer if ctype == LEFT else -max_steer if ctype == RIGHT else 0.0
        for index in range(1, steps + 1):
            travel_m = direction * min(segment_m, index * segment_m / float(steps))
            px, py, pyaw = _interpolate_local(
                ox,
                oy,
                oyaw,
                travel_m * max_curvature,
                ctype,
                max_curvature,
            )
            local_x.append(px)
            local_y.append(py)
            local_yaw.append(pyaw)
            directions.append(direction)
            steers.append(steer)

    c = math.cos(syaw)
    s = math.sin(syaw)
    path.x = [c * x - s * y + sx for x, y in zip(local_x, local_y)]
    path.y = [s * x + c * y + sy for x, y in zip(local_x, local_y)]
    path.yaw = [wrap_pi(value + syaw) for value in local_yaw]
    path.directions = directions
    path.steers = steers
    return path


def calc_paths(
    start: Sequence[float],
    goal: Sequence[float],
    max_curvature: float,
    step_size: float = 0.10,
    max_length_m: float = 80.0,
    max_steer: float = 0.0,
) -> List[ReedsSheppPath]:
    paths = generate_paths(start, goal, max_curvature, max_length_m=max_length_m)
    sampled = [
        sample_path(path, start, max_curvature, step_size=step_size, max_steer=max_steer)
        for path in paths
    ]
    # Numerical/domain edge cases in a canonical word must never leak into an
    # analytic expansion.  Keep only paths that actually terminate at q1.
    gx, gy, gyaw = map(float, goal[:3])
    return [
        path
        for path in sampled
        if path.x
        and math.hypot(path.x[-1] - gx, path.y[-1] - gy) <= 1e-5
        and abs(wrap_pi(path.yaw[-1] - gyaw)) <= 1e-5
    ]


def path_cost(
    path: ReedsSheppPath,
    reverse_multiplier: float = 1.25,
    gear_switch_cost: float = 2.0,
    steer_cost: float = 0.10,
    steer_change_cost: float = 0.25,
) -> float:
    cost = 0.0
    controls: List[float] = []
    for length, ctype in zip(path.lengths, path.ctypes):
        cost += abs(length) * (1.0 if length >= 0.0 else float(reverse_multiplier))
        control = 0.0 if ctype == STRAIGHT else 1.0 if ctype == LEFT else -1.0
        controls.append(control)
        cost += abs(length) * float(steer_cost) * abs(control)
    for a, b in zip(path.lengths, path.lengths[1:]):
        if a * b < 0.0:
            cost += float(gear_switch_cost)
    for a, b in zip(controls, controls[1:]):
        cost += float(steer_change_cost) * abs(b - a)
    return float(cost)


def sorted_paths(paths: Iterable[ReedsSheppPath], **cost_kwargs: float) -> List[ReedsSheppPath]:
    return sorted(paths, key=lambda item: path_cost(item, **cost_kwargs))
