"""Differentiable decoder and dimensionless auxiliary losses.

This training approximation never replaces the float64 independent validator.
"""
import numpy as np
import torch
from torch.nn import functional as F
from .config import DEFAULT


def decode_physics(latent, hclass, cfg=DEFAULT):
    # Double precision prevents cancellation on tiny cosine-grid intervals.
    a = latent.double()
    batch, phases, knots, _ = a.shape
    phase_index = torch.arange(phases, device=a.device)
    valid = phase_index[None] < (hclass % cfg.max_phases+1)[:, None]
    gear = (torch.where(hclass < cfg.max_phases, 1., -1.)[:, None]*torch.where(phase_index % 2 == 0, 1., -1.)[None]).to(a.dtype)
    u = .5*(1-torch.cos(torch.linspace(0, np.pi, knots, device=a.device, dtype=a.dtype)))
    # Training stabilization only; strict inference decode rejects out-of-scope lengths.
    length = torch.exp(a[:, :, 0, 0].clamp(-12, np.log(cfg.max_total_length)))*valid
    delta = cfg.max_steer*torch.tanh(a[..., 1])
    b = cfg.max_speed**2*torch.sigmoid(a[..., 2])
    interior = torch.ones(knots, device=a.device, dtype=a.dtype)
    interior[[0, -1]] = 0
    b = b*interior*valid[..., None]
    distance = length[..., None]*torch.diff(u)
    difference = torch.diff(delta, dim=-1)
    base = delta[..., :-1]

    def turn(t):
        t = t.reshape(1, 1, 1, -1)
        dx = difference[..., None]*t
        varying = difference.abs() > 1e-7
        safe = torch.where(varying, difference, torch.ones_like(difference))
        logarithm = torch.log1p(-2*torch.sin(dx/2)**2-torch.tan(base[..., None])*torch.sin(dx))
        general = -distance[..., None]/safe[..., None]*logarithm
        nearly_constant = distance[..., None]*t*torch.tan(base[..., None]+.5*dx)
        return gear[..., None, None]/cfg.wheelbase*torch.where(varying[..., None], general, nearly_constant)

    turning = turn(torch.ones(1, device=a.device, dtype=a.dtype))[..., 0]
    flat_turn = turning.reshape(batch, -1)
    yaw_before = F.pad(torch.cumsum(flat_turn, dim=-1)[:, :-1], (1, 0)).reshape_as(turning)
    nodes, weights = np.polynomial.legendre.leggauss(8)
    nodes = torch.as_tensor((nodes+1)/2, device=a.device, dtype=a.dtype)
    weights = torch.as_tensor(weights/2, device=a.device, dtype=a.dtype)
    theta = yaw_before[..., None]+turn(nodes)
    dx = gear[..., None]*distance*(theta.cos()*weights).sum(-1)
    dy = gear[..., None]*distance*(theta.sin()*weights).sum(-1)
    flat_x = torch.cumsum(dx.reshape(batch, -1), dim=-1)
    flat_y = torch.cumsum(dy.reshape(batch, -1), dim=-1)
    flat_yaw = torch.cumsum(flat_turn, dim=-1)
    # Nodes at phase starts are copied from the preceding integrated endpoint.
    ends = torch.stack((flat_x, flat_y, flat_yaw), dim=-1).reshape(batch, phases, knots-1, 3)
    starts = torch.cat((torch.zeros((batch, 1, 1, 3), device=a.device, dtype=a.dtype), ends[:, :-1, -1:, :]), dim=1)
    poses = torch.cat((starts, ends), dim=2)
    q = torch.sqrt(b.clamp_min(1e-20))*interior*valid[..., None]
    q0, q1 = q[..., :-1], q[..., 1:]
    safe_distance = distance.clamp_min(1e-12)
    dt = 2*distance/(q0+q1).clamp_min(1e-12)
    acceleration = torch.diff(b, dim=-1)/(2*safe_distance)
    slope = difference/safe_distance*valid[..., None]
    rate = slope.abs()*torch.maximum(q0, q1)
    f4 = (q0**4+q0**3*q1+q0*q0*q1*q1+q0*q1**3+q1**4)/5
    ac = (dt*acceleration.square()).sum((1, 2))
    sc = (dt*slope.square()*f4).sum((1, 2))
    # Dwell at valid boundaries only; first/final steering are zero in current scope.
    start_dwell = delta[:, 0, 0].abs()
    middle = ((delta[:, 1:, 0]-delta[:, :-1, -1]).abs()*valid[:, 1:]).sum(1)
    last_index = (hclass % cfg.max_phases).long()
    final_dwell = delta[torch.arange(batch, device=a.device), last_index, -1].abs()
    dwell = (start_dwell+middle+final_dwell)/(cfg.max_steer_rate-cfg.control_margin)
    duration = dt.sum((1, 2))+dwell
    return dict(lengths=length, delta=delta, b=b, poses=poses, endpoint=ends[:, -1, -1],
                valid=valid, acceleration=acceleration, steer_rate=rate,
                J=duration+.5*(ac+sc), duration_s=duration)


def normalized_cost_loss(generated_j, teacher_j):
    """Direct J objective, including below-teacher and initially infeasible paths.

    Feasibility is trained by the separate constraint losses and enforced by
    deployment validation. Teacher costs only set a detached, per-scene scale.
    """
    return (generated_j/teacher_j.detach().to(generated_j).clamp_min(1e-6)).mean()


def auxiliary_losses(latent, h, batch, cfg=DEFAULT, cost_objective='legacy_excess'):
    values = decode_physics(latent, h, cfg)
    end = values['endpoint']
    goal = batch['goal'].to(end)
    yaw = torch.atan2(torch.sin(end[:, 2]-goal[:, 2]), torch.cos(end[:, 2]-goal[:, 2]))
    # Smooth L1 avoids a few gross early predictions dominating every gradient.
    position_loss = F.smooth_l1_loss((end[:, :2]-goal[:, :2])/.1, torch.zeros_like(end[:, :2]))
    yaw_loss = F.smooth_l1_loss(yaw/.05, torch.zeros_like(yaw))
    endpoint = position_loss+yaw_loss
    valid = values['valid']
    interval_mask = valid[..., None].expand_as(values['acceleration'])
    av = F.relu(values['acceleration'].abs()/cfg.max_accel-1)
    sv = F.relu(values['steer_rate']/cfg.max_steer_rate-1)
    # Quadratic near the boundary, linear on gross early violations: the tiny
    # end intervals must not make random initial rollouts dominate all learning.
    acceleration = F.smooth_l1_loss(av, torch.zeros_like(av), reduction='none')
    steering = F.smooth_l1_loss(sv, torch.zeros_like(sv), reduction='none')
    limits = ((acceleration+steering)*interval_mask).sum()/interval_mask.sum().clamp_min(1)
    limits = limits+F.relu(values['lengths'].sum(1)/cfg.max_total_length-1).square().mean()
    # Every eighth control node plus last node; full geometry is checked later.
    indices = torch.unique(torch.cat((torch.arange(0, cfg.intervals+1, 8, device=end.device),
                                      torch.tensor([cfg.intervals], device=end.device))))
    poses = values['poses'][:, :, indices].float()
    bx = torch.linspace(-.98025278, 3.81152672, 7, device=end.device)
    by = torch.linspace(-1.081725, 1.081725, 5, device=end.device)
    body = torch.cartesian_prod(bx, by)
    c, s = poses[..., 2].cos(), poses[..., 2].sin()
    x = poses[..., 0, None]+c[..., None]*body[:, 0]-s[..., None]*body[:, 1]
    y = poses[..., 1, None]+s[..., None]*body[:, 0]+c[..., None]*body[:, 1]
    start = batch['scene_start'].float()
    cs, ss = start[:, 2].cos()[:, None, None, None], start[:, 2].sin()[:, None, None, None]
    source_x = start[:, 0, None, None, None]+cs*x-ss*y
    source_y = start[:, 1, None, None, None]+ss*x+cs*y
    # Raw raster rows run from +x to -x, columns from -y to +y.
    sample_grid = torch.stack((source_y/10, -source_x/10), dim=-1)
    sdf = F.grid_sample(batch['sdf'].float()[:, None], sample_grid.reshape(len(end), -1, 1, 2),
                        mode='bilinear', padding_mode='border', align_corners=False)[:, 0, :, 0]
    outside = F.relu(sample_grid.abs()-1).amax(-1).reshape(len(end), -1)*10
    sdf = sdf-outside
    collision_mask = valid[:, :, None, None].expand_as(x).reshape(len(end), -1)
    violation = F.relu((cfg.clearance-sdf)/.1)
    # Mean plus worst violation keeps short collisions from disappearing.
    collision = (violation.square()*collision_mask).sum()/collision_mask.sum().clamp_min(1)
    collision = collision+(violation*collision_mask).amax(1).square().mean()
    teacher_J = batch['target_J'].to(end)
    position_error = (end[:, :2]-goal[:, :2]).norm(dim=-1)
    physical_gate = (values['steer_rate'].amax((1, 2)) <= cfg.max_steer_rate)
    physical_gate &= (values['acceleration'].abs().amax((1, 2)) <= cfg.max_accel)
    physical_gate &= ((sdf.masked_fill(~collision_mask, float('inf'))).amin(1) >= cfg.clearance)
    sampled_feasible = physical_gate & (position_error < cfg.terminal_position_tolerance_m)
    sampled_feasible &= yaw.abs() < cfg.terminal_yaw_tolerance_rad
    if cost_objective == 'normalized_j':
        j_loss = normalized_cost_loss(values['J'], teacher_J)
        cost_samples = torch.ones_like(position_error)
    elif cost_objective == 'legacy_excess':
        # Retained explicitly for reproducibility of older training commands.
        feasible_gate = physical_gate & (position_error < .05) & (yaw.abs() < .02)
        excess = F.relu(values['J']/teacher_J.clamp_min(1)-1)
        j_loss = (excess*feasible_gate.detach()).sum()/feasible_gate.sum().clamp_min(1)
        cost_samples = feasible_gate.to(end.dtype)
    else:
        raise ValueError('unknown cost objective: '+cost_objective)
    return dict(endpoint=endpoint.float(), endpoint_position=position_loss.float(),endpoint_yaw=yaw_loss.float(),
                endpoint_m=(end[:,:2]-goal[:,:2]).norm(dim=-1).mean().float(),yaw_rad=yaw.abs().mean().float(),
                limits=limits.float(), collision=collision.float(),
                cost=j_loss.float(),cost_sample_fraction=cost_samples.mean().float(),
                sampled_feasible_fraction=sampled_feasible.float().mean(),
                generated_J=values['J'].mean().float(),
                generated_J_ratio=(values['J']/teacher_J.clamp_min(1e-6)).mean().float(),
                generated_duration_s=values['duration_s'].mean().float()), values
