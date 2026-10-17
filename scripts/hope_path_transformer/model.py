"""ConvNeXt-FPN/vector, multi-candidate coarse-to-fine path generator."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .config import HopePathConfig
from .geometry import CURVATURE_DISCRETIZATION_MARGIN


class _MicroBackbone(nn.Module):
    """Small dependency-light backbone used only by unit/smoke tests."""

    channels = (32, 64, 128, 256)

    def __init__(self, input_channels: int):
        super().__init__()
        stages = []
        incoming = input_channels
        for outgoing in self.channels:
            stages.append(
                nn.Sequential(
                    nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1),
                    nn.GroupNorm(max(1, outgoing // 16), outgoing),
                    nn.GELU(),
                    nn.Conv2d(outgoing, outgoing, 3, padding=1, groups=outgoing),
                    nn.GELU(),
                )
            )
            incoming = outgoing
        self.stages = nn.ModuleList(stages)

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for stage in self.stages:
            value = stage(value)
            outputs.append(value)
        return outputs


class ConvNeXtFeatureBackbone(nn.Module):
    channels = (96, 192, 384, 768)

    def __init__(self, input_channels: int):
        super().__init__()
        try:
            from torchvision.models import convnext_tiny
        except ImportError as error:  # pragma: no cover - dependency message
            raise RuntimeError("torchvision is required for ConvNeXt-Tiny") from error
        network = convnext_tiny(weights=None)
        old = network.features[0][0]
        network.features[0][0] = nn.Conv2d(
            input_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
        self.features = network.features

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for index, module in enumerate(self.features):
            value = module(value)
            if index in (1, 3, 5, 7):
                outputs.append(value)
        return outputs


class FeaturePyramid(nn.Module):
    def __init__(self, input_channels: Sequence[int], hidden_dim: int):
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(channels, hidden_dim, 1) for channels in input_channels]
        )
        self.output = nn.ModuleList(
            [nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1) for _ in input_channels]
        )

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        lateral = [layer(value) for layer, value in zip(self.lateral, features)]
        for index in range(len(lateral) - 2, -1, -1):
            lateral[index] = lateral[index] + F.interpolate(
                lateral[index + 1], size=lateral[index].shape[-2:], mode="nearest"
            )
        return [layer(value) for layer, value in zip(self.output, lateral)]


class RasterEncoder(nn.Module):
    def __init__(self, config: HopePathConfig):
        super().__init__()
        if config.model.backbone == "convnext_tiny":
            self.backbone = ConvNeXtFeatureBackbone(config.model.raster_channels)
        elif config.model.backbone == "micro":
            self.backbone = _MicroBackbone(config.model.raster_channels)
        else:
            raise ValueError("unsupported backbone: {}".format(config.model.backbone))
        self.fpn = FeaturePyramid(self.backbone.channels, config.model.hidden_dim)

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        return self.fpn(self.backbone(value.float()))


class PolygonVectorEncoder(nn.Module):
    def __init__(self, config: HopePathConfig):
        super().__init__()
        dim = config.model.hidden_dim
        self.half_extent = 0.5 * config.data.extent_m
        self.vertex = nn.Sequential(
            nn.Linear(2, dim // 2), nn.LayerNorm(dim // 2), nn.GELU(),
            nn.Linear(dim // 2, dim), nn.GELU(),
        )
        self.goal = nn.Sequential(
            nn.Linear(4, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.type_embedding = nn.Parameter(torch.zeros(1, 2, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.model.attention_heads,
            dim_feedforward=4 * dim,
            dropout=config.model.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.model.vector_layers, enable_nested_tensor=False
        )
        self.normalization = nn.LayerNorm(dim)

    def forward(
        self,
        vertices: torch.Tensor,
        vertex_mask: torch.Tensor,
        polygon_mask: torch.Tensor,
        goal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_vertices = vertices / self.half_extent
        encoded = self.vertex(normalized_vertices)
        weights = vertex_mask.unsqueeze(-1).to(encoded.dtype)
        polygon = (encoded * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        normalized_goal = torch.cat(
            (goal[:, :2] / self.half_extent, goal[:, 2:4]), dim=-1
        )
        goal_token = self.goal(normalized_goal).unsqueeze(1) + self.type_embedding[:, :1]
        polygon = polygon + self.type_embedding[:, 1:2]
        memory = torch.cat((goal_token, polygon), dim=1)
        valid = torch.cat(
            (torch.ones((len(goal), 1), dtype=torch.bool, device=goal.device), polygon_mask),
            dim=1,
        )
        memory = self.transformer(memory, src_key_padding_mask=~valid)
        return self.normalization(memory), valid


class MultiScaleDeformableAttention(nn.Module):
    """Pure-PyTorch multi-scale point sampling around reference coordinates."""

    def __init__(self, dim: int, heads: int, levels: int, points: int):
        super().__init__()
        if dim % heads:
            raise ValueError("attention dimension must be divisible by heads")
        self.dim, self.heads = dim, heads
        self.levels, self.points = levels, points
        self.head_dim = dim // heads
        self.offsets = nn.Linear(dim, heads * levels * points * 2)
        self.weights = nn.Linear(dim, heads * levels * points)
        self.output = nn.Linear(dim, dim)
        nn.init.zeros_(self.offsets.weight)
        angles = torch.arange(heads, dtype=torch.float32) * (2.0 * math.pi / heads)
        direction = torch.stack((angles.cos(), angles.sin()), dim=-1)
        direction = direction[:, None, None, :].repeat(1, levels, points, 1)
        direction = direction * torch.arange(1, points + 1, dtype=torch.float32)[None, None, :, None]
        with torch.no_grad():
            self.offsets.bias.copy_(direction.reshape(-1))
        nn.init.zeros_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)

    def forward(
        self,
        query: torch.Tensor,
        references: torch.Tensor,
        features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        batch, queries, _ = query.shape
        offsets = self.offsets(query).reshape(
            batch, queries, self.heads, self.levels, self.points, 2
        )
        attention = torch.softmax(
            self.weights(query).reshape(batch, queries, self.heads, -1), dim=-1
        ).reshape(batch, queries, self.heads, self.levels, self.points)
        accumulated = query.new_zeros((batch, self.heads, self.head_dim, queries))
        for level, feature in enumerate(features):
            height, width = feature.shape[-2:]
            value = feature.reshape(batch, self.heads, self.head_dim, height, width)
            value = value.reshape(batch * self.heads, self.head_dim, height, width)
            scale = query.new_tensor((width, height)).view(1, 1, 1, 1, 2)
            grid = references[:, :, None, None, :] + offsets[:, :, :, level] / scale
            grid = grid.permute(0, 2, 1, 3, 4).reshape(
                batch * self.heads, queries, self.points, 2
            )
            sampled = F.grid_sample(
                value, grid.mul(2.0).sub(1.0), mode="bilinear",
                padding_mode="zeros", align_corners=False,
            ).reshape(batch, self.heads, self.head_dim, queries, self.points)
            level_weight = attention[:, :, :, level].permute(0, 2, 1, 3).unsqueeze(2)
            accumulated = accumulated + (sampled * level_weight).sum(dim=-1)
        merged = accumulated.permute(0, 3, 1, 2).reshape(batch, queries, self.dim)
        return self.output(merged)


class PathDecoderLayer(nn.Module):
    def __init__(self, config: HopePathConfig):
        super().__init__()
        dim, heads = config.model.hidden_dim, config.model.attention_heads
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            dim, heads, dropout=config.model.dropout, batch_first=True
        )
        self.vector_norm = nn.LayerNorm(dim)
        self.vector_attention = nn.MultiheadAttention(
            dim, heads, dropout=config.model.dropout, batch_first=True
        )
        self.raster_norm = nn.LayerNorm(dim)
        self.raster_attention = MultiScaleDeformableAttention(
            dim, heads, levels=4, points=config.model.deformable_points
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(config.model.dropout),
            nn.Linear(4 * dim, dim),
        )
        self.reference_delta = nn.Linear(dim, 2)

    def forward(
        self,
        query: torch.Tensor,
        references: torch.Tensor,
        raster: Sequence[torch.Tensor],
        vector_memory: torch.Tensor,
        vector_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, candidates, steps, dim = query.shape
        separated = query.reshape(batch * candidates, steps, dim)
        normalized = self.self_norm(separated)
        attended, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        separated = separated + attended
        memory = vector_memory[:, None].expand(-1, candidates, -1, -1).reshape(
            batch * candidates, vector_memory.shape[1], dim
        )
        valid = vector_valid[:, None].expand(-1, candidates, -1).reshape(
            batch * candidates, vector_valid.shape[1]
        )
        normalized = self.vector_norm(separated)
        attended, _ = self.vector_attention(
            normalized, memory, memory, key_padding_mask=~valid, need_weights=False
        )
        separated = separated + attended
        query = separated.reshape(batch, candidates, steps, dim)
        flattened = query.reshape(batch, candidates * steps, dim)
        reference_flat = references.reshape(batch, candidates * steps, 2)
        flattened = flattened + self.raster_attention(
            self.raster_norm(flattened), reference_flat, raster
        )
        flattened = flattened + self.ffn(self.ffn_norm(flattened))
        references = (
            reference_flat + 0.05 * torch.tanh(self.reference_delta(flattened))
        ).clamp(0.0, 1.0).reshape(batch, candidates, steps, 2)
        return flattened.reshape(batch, candidates, steps, dim), references


def integrate_primitives(
    curvature: torch.Tensor,
    direction: torch.Tensor,
    step_m: float,
    step_magnitude: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable midpoint bicycle integration with pose 0 fixed at origin."""

    # Each segment only depends on the cumulative yaw of the preceding
    # segments.  Expressing the full rollout with cumulative sums avoids 191
    # tiny Python-dispatched GPU kernels for a 192-primitive path while
    # preserving the midpoint integration used by the original loop.
    motion_curvature = curvature[..., :-1]
    motion_direction = direction[..., :-1]
    if step_magnitude is None:
        motion_step = torch.full_like(motion_curvature, float(step_m))
    else:
        motion_step = step_magnitude[..., :-1]
    signed_step = motion_step * motion_direction
    yaw_delta = signed_step * motion_curvature
    yaw_after_unwrapped = torch.cumsum(yaw_delta, dim=-1)
    zero = torch.zeros_like(yaw_after_unwrapped[..., :1])
    yaw_before = torch.cat((zero, yaw_after_unwrapped[..., :-1]), dim=-1)
    midpoint = yaw_before + 0.5 * yaw_delta
    x = torch.cat((zero, torch.cumsum(signed_step * torch.cos(midpoint), dim=-1)), dim=-1)
    y = torch.cat((zero, torch.cumsum(signed_step * torch.sin(midpoint), dim=-1)), dim=-1)
    yaw = torch.cat(
        (
            zero,
            torch.atan2(
                torch.sin(yaw_after_unwrapped), torch.cos(yaw_after_unwrapped)
            ),
        ),
        dim=-1,
    )
    return torch.stack((x, y, yaw), dim=-1)


def anchor_terminal_pose_trace(
    poses: torch.Tensor,
    goal: torch.Tensor,
    lengths: torch.Tensor,
    bridge_steps: int,
) -> torch.Tensor:
    """Smoothly impose the exact rear-axle goal pose on every candidate.

    Only the final ``bridge_steps`` intervals are corrected. A quintic blend
    leaves the learned prefix untouched, has zero correction slope at the
    splice and goal, and holds padded tokens at the goal instead of creating a
    single impossible final jump.
    """

    if poses.ndim != 4 or poses.shape[-1] != 3:
        raise ValueError("poses must have shape [B,K,T,3]")
    if goal.ndim != 2 or goal.shape[-1] != 4 or len(goal) != len(poses):
        raise ValueError("goal must have shape [B,4]")
    if int(bridge_steps) < 2:
        raise ValueError("bridge_steps must be at least two")
    batch, candidates, steps, _ = poses.shape
    if lengths.ndim == 1:
        lengths = lengths[:, None].expand(-1, candidates)
    if lengths.shape != (batch, candidates):
        raise ValueError("lengths must have shape [B] or [B,K]")
    lengths = lengths.to(device=poses.device, dtype=torch.long).clamp(2, steps)
    endpoint_index = lengths - 1
    endpoint = poses.gather(
        2, endpoint_index[..., None, None].expand(-1, -1, 1, 3)
    ).squeeze(2)
    goal_yaw = torch.atan2(goal[:, 2], goal[:, 3])
    goal_pose = torch.cat((goal[:, :2], goal_yaw[:, None]), dim=-1)
    goal_pose = goal_pose[:, None].expand(-1, candidates, -1)

    bridge_begin = (endpoint_index - int(bridge_steps)).clamp_min(1)
    token = torch.arange(steps, device=poses.device).view(1, 1, steps)
    denominator = (endpoint_index - bridge_begin).clamp_min(1).to(poses.dtype)
    progress = (
        (token - bridge_begin[..., None]).to(poses.dtype)
        / denominator[..., None]
    ).clamp(0.0, 1.0)
    weight = progress.pow(3) * (
        10.0 - 15.0 * progress + 6.0 * progress.square()
    )
    xy_error = goal_pose[..., :2] - endpoint[..., :2]
    yaw_error = torch.atan2(
        torch.sin(goal_pose[..., 2] - endpoint[..., 2]),
        torch.cos(goal_pose[..., 2] - endpoint[..., 2]),
    )
    corrected_xy = poses[..., :2] + weight[..., None] * xy_error[..., None, :]
    corrected_yaw = poses[..., 2] + weight * yaw_error[..., None]
    corrected_yaw = torch.atan2(torch.sin(corrected_yaw), torch.cos(corrected_yaw))
    corrected = torch.cat((corrected_xy, corrected_yaw[..., None]), dim=-1)
    goal_rows = goal_pose[..., None, :].expand(-1, -1, steps, -1)
    after_endpoint = token > endpoint_index[..., None]
    corrected = torch.where(after_endpoint[..., None], goal_rows, corrected)
    at_endpoint = token == endpoint_index[..., None]
    return torch.where(at_endpoint[..., None], goal_rows, corrected)


def terminal_pose_error(
    poses: torch.Tensor,
    goal: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return candidate endpoint position and absolute yaw errors."""

    batch, candidates, steps, _ = poses.shape
    if lengths.ndim == 1:
        lengths = lengths[:, None].expand(-1, candidates)
    lengths = lengths.to(device=poses.device, dtype=torch.long).clamp(2, steps)
    endpoint = poses.gather(
        2, (lengths - 1)[..., None, None].expand(-1, -1, 1, 3)
    ).squeeze(2)
    goal_yaw = torch.atan2(goal[:, 2], goal[:, 3])[:, None]
    position = torch.linalg.vector_norm(
        endpoint[..., :2] - goal[:, None, :2], dim=-1
    )
    yaw = torch.abs(
        torch.atan2(
            torch.sin(endpoint[..., 2] - goal_yaw),
            torch.cos(endpoint[..., 2] - goal_yaw),
        )
    )
    return position, yaw


def controls_from_pose_trace(
    poses: torch.Tensor,
    direction: torch.Tensor,
    lengths: torch.Tensor,
    wheelbase_m: float,
) -> dict[str, torch.Tensor]:
    """Recompute path controls after the hard terminal bridge."""

    if poses.ndim != 4 or poses.shape[-1] != 3:
        raise ValueError("poses must have shape [B,K,T,3]")
    if direction.shape != poses.shape[:-1]:
        raise ValueError("direction must have shape [B,K,T]")
    if float(wheelbase_m) <= 0.0:
        raise ValueError("wheelbase_m must be positive")
    batch, candidates, steps, _ = poses.shape
    if lengths.ndim == 1:
        lengths = lengths[:, None].expand(-1, candidates)
    lengths = lengths.to(device=poses.device, dtype=torch.long).clamp(2, steps)
    if lengths.shape != (batch, candidates):
        raise ValueError("lengths must have shape [B] or [B,K]")

    delta_xy = poses[..., 1:, :2] - poses[..., :-1, :2]
    edge_step = torch.linalg.vector_norm(delta_xy, dim=-1)
    yaw_delta = torch.atan2(
        torch.sin(poses[..., 1:, 2] - poses[..., :-1, 2]),
        torch.cos(poses[..., 1:, 2] - poses[..., :-1, 2]),
    )
    direction_sign = torch.where(
        direction[..., :-1] >= 0.0,
        torch.ones_like(direction[..., :-1]),
        -torch.ones_like(direction[..., :-1]),
    )
    edge_active = (
        torch.arange(steps - 1, device=poses.device).view(1, 1, -1)
        < (lengths - 1)[..., None]
    )
    signed_step = direction_sign * edge_step.clamp_min(1.0e-8)
    edge_curvature = torch.where(
        edge_active & (edge_step > 1.0e-8),
        yaw_delta / signed_step,
        torch.zeros_like(yaw_delta),
    )
    edge_step = torch.where(edge_active, edge_step, torch.zeros_like(edge_step))
    zero = torch.zeros_like(edge_curvature[..., :1])
    curvature = torch.cat((edge_curvature, zero), dim=-1)
    step_magnitude = torch.cat((edge_step, zero), dim=-1)
    steering = torch.atan(float(wheelbase_m) * curvature)
    steering_increment = torch.cat(
        (steering[..., :1], torch.diff(steering, dim=-1)), dim=-1
    )
    return {
        "curvature": curvature,
        "steering": steering,
        "steering_increment": steering_increment,
        "step_magnitude": step_magnitude,
        "raw_step_magnitude": step_magnitude,
    }


def densify_control_rollout(
    steering: torch.Tensor,
    direction: torch.Tensor,
    step_magnitude: torch.Tensor,
    *,
    wheelbase_m: float,
    substeps: int,
) -> dict[str, torch.Tensor]:
    """Expand coarse controls into a differentiable controller-rate path.

    Transformer attention remains at ``T`` tokens.  Each of its ``T-1``
    motion intervals is integrated as ``substeps`` bicycle-model intervals,
    while steering is linearly interpolated between neighboring control
    knots.  This avoids the quadratic memory cost of increasing decoder token
    count and still exposes the approximately 5 cm path consumed by CARLA.
    """

    substeps = int(substeps)
    if substeps < 1:
        raise ValueError("substeps must be positive")
    if steering.shape != direction.shape or steering.shape != step_magnitude.shape:
        raise ValueError("steering, direction, and step_magnitude shapes must match")
    if substeps == 1:
        curvature = torch.tan(steering) / float(wheelbase_m)
        return {
            "poses": integrate_primitives(
                curvature, direction, 0.0, step_magnitude
            ),
            "steering": steering,
            "curvature": curvature,
            "direction": direction,
            "step_magnitude": step_magnitude * direction.abs(),
            "raw_step_magnitude": step_magnitude,
        }

    dtype, device = steering.dtype, steering.device
    node_fraction = torch.arange(substeps, dtype=dtype, device=device) / float(
        substeps
    )
    steering_delta = steering[..., 1:] - steering[..., :-1]
    dense_steering = (
        steering[..., :-1, None]
        + steering_delta[..., None] * node_fraction
    ).flatten(start_dim=-2)
    dense_steering = torch.cat((dense_steering, steering[..., -1:]), dim=-1)

    dense_curvature = torch.tan(dense_steering) / float(wheelbase_m)

    dense_direction = direction[..., :-1, None].expand(
        *direction.shape[:-1], direction.shape[-1] - 1, substeps
    ).flatten(start_dim=-2)
    dense_direction = torch.cat((dense_direction, direction[..., -1:]), dim=-1)
    dense_raw_step = (
        step_magnitude[..., :-1, None]
        .expand(*step_magnitude.shape[:-1], step_magnitude.shape[-1] - 1, substeps)
        .flatten(start_dim=-2)
        / float(substeps)
    )
    dense_raw_step = torch.cat(
        (dense_raw_step, torch.zeros_like(dense_raw_step[..., -1:])), dim=-1
    )
    dense_poses = integrate_primitives(
        dense_curvature, dense_direction, 0.0, dense_raw_step
    )
    return {
        "poses": dense_poses,
        "steering": dense_steering,
        "curvature": dense_curvature,
        "direction": dense_direction,
        "step_magnitude": dense_raw_step * dense_direction.abs(),
        "raw_step_magnitude": dense_raw_step,
    }


class HopePathPlanner(nn.Module):
    """Generate K complete paths in one coarse-to-fine forward pass."""

    def __init__(self, config: HopePathConfig):
        super().__init__()
        config.validate()
        self.config = config
        dim = config.model.hidden_dim
        self.raster = RasterEncoder(config)
        self.vector = PolygonVectorEncoder(config)
        self.mode_embedding = nn.Embedding(config.model.candidate_count, dim)
        self.start_direction_embedding = (
            nn.Embedding(2, dim) if config.model.condition_start_direction else None
        )
        self.coarse_progress = nn.Embedding(config.data.coarse_count, dim)
        self.refine_progress = nn.Embedding(config.data.primitive_count, dim)
        self.goal_context = nn.Sequential(nn.Linear(4, dim), nn.LayerNorm(dim), nn.GELU())
        self.coarse_decoder = nn.ModuleList(
            [PathDecoderLayer(config) for _ in range(config.model.coarse_layers)]
        )
        self.coarse_yaw = nn.Linear(dim, 2)
        self.refinement_decoder = nn.ModuleList(
            [PathDecoderLayer(config) for _ in range(config.model.refinement_layers)]
        )
        self.curvature = nn.Linear(dim, 1)
        if config.model.bounded_steering_increments:
            # Start a fresh incremental decoder from a straight, controllable
            # trajectory rather than a 192-step random walk into saturation.
            nn.init.zeros_(self.curvature.weight)
            nn.init.zeros_(self.curvature.bias)
        self.step = nn.Linear(dim, 1)
        self.gear = nn.Linear(dim, 2)
        self.length = nn.Linear(dim, 1)
        self.score = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1)
        )

    def _initial_queries(
        self,
        goal: torch.Tensor,
        candidates: int,
        start_direction: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = len(goal)
        coarse_count = self.config.data.coarse_count
        progress = torch.linspace(0.0, 1.0, coarse_count, device=goal.device, dtype=goal.dtype)
        data = self.config.data
        start = goal.new_tensor(
            (
                0.5 - data.raster_center_x_m / data.extent_m,
                0.5 + data.raster_center_y_m / data.extent_m,
            )
        ).view(1, 1, 2)
        goal_reference = torch.stack(
            (
                (goal[:, 0] - data.raster_center_x_m) / data.extent_m + 0.5,
                0.5 - (goal[:, 1] - data.raster_center_y_m) / data.extent_m,
            ),
            dim=-1,
        ).clamp(0.0, 1.0).unsqueeze(1)
        reference = start + progress.view(1, -1, 1) * (goal_reference - start)
        reference = reference[:, None].expand(-1, candidates, -1, -1).contiguous()
        query = self.coarse_progress.weight[None, None].expand(batch, candidates, -1, -1)
        query = query + self.mode_embedding.weight[None, :candidates, None]
        query = query + self.goal_context(goal)[:, None, None]
        if self.start_direction_embedding is not None:
            if start_direction is None:
                raise KeyError(
                    "start_direction is required when condition_start_direction is enabled"
                )
            direction_index = (start_direction.reshape(batch) > 0).long()
            direction_context = self.start_direction_embedding(direction_index)
            query = query + direction_context[:, None, None]
        return query, reference

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        candidate_count: int | None = None,
        *,
        use_batch_terminal_lengths: bool = True,
    ) -> dict[str, torch.Tensor]:
        candidates = int(candidate_count or self.config.model.candidate_count)
        if not 1 <= candidates <= self.config.model.candidate_count:
            raise ValueError("candidate_count is outside configured range")
        raster = self.raster(batch["bev"])
        vector_memory, vector_valid = self.vector(
            batch["obstacle_vertices"], batch["obstacle_vertex_mask"],
            batch["obstacle_mask"], batch["goal"],
        )
        query, reference = self._initial_queries(
            batch["goal"], candidates, batch.get("start_direction")
        )
        for layer in self.coarse_decoder:
            query, reference = layer(query, reference, raster, vector_memory, vector_valid)

        coarse_xy = torch.stack(
            (
                (reference[..., 0] - 0.5) * self.config.data.extent_m
                + self.config.data.raster_center_x_m,
                (0.5 - reference[..., 1]) * self.config.data.extent_m
                + self.config.data.raster_center_y_m,
            ),
            dim=-1,
        )
        yaw_vector = self.coarse_yaw(query)
        coarse_yaw = torch.atan2(yaw_vector[..., 0], yaw_vector[..., 1])
        coarse_pose = torch.cat((coarse_xy, coarse_yaw.unsqueeze(-1)), dim=-1)

        batch_size, _, coarse_steps, dim = query.shape
        primitive_count = self.config.data.primitive_count
        query = F.interpolate(
            query.reshape(batch_size * candidates, coarse_steps, dim).transpose(1, 2),
            size=primitive_count, mode="linear", align_corners=True,
        ).transpose(1, 2).reshape(batch_size, candidates, primitive_count, dim)
        reference = F.interpolate(
            reference.reshape(batch_size * candidates, coarse_steps, 2).transpose(1, 2),
            size=primitive_count, mode="linear", align_corners=True,
        ).transpose(1, 2).reshape(batch_size, candidates, primitive_count, 2)
        query = query + self.refine_progress.weight[None, None]
        for layer in self.refinement_decoder:
            query, reference = layer(query, reference, raster, vector_memory, vector_valid)

        direct_curvature = (
            torch.tanh(self.curvature(query).squeeze(-1))
            * self.config.vehicle.max_curvature_m_inv
            * CURVATURE_DISCRETIZATION_MARGIN
        )
        if self.config.model.bounded_steering_increments:
            max_delta = math.radians(self.config.model.max_steering_delta_deg)
            proposed_delta = torch.tanh(self.curvature(query).squeeze(-1)) * max_delta
            zero_delta = torch.zeros_like(proposed_delta[..., :1])
            proposed_delta = torch.cat((zero_delta, proposed_delta[..., 1:]), dim=-1)
            start_steering = batch.get("start_steering_rad")
            if start_steering is None:
                start_steering = torch.zeros_like(proposed_delta[..., :1])
            else:
                start_steering = start_steering.to(
                    device=query.device, dtype=query.dtype
                ).reshape(batch_size, 1, 1)
            maximum_steering = math.atan(
                self.config.vehicle.wheelbase_m
                * self.config.vehicle.max_curvature_m_inv
                * CURVATURE_DISCRETIZATION_MARGIN
            )
            # Smooth saturation keeps gradients alive near the rack limit.
            # tanh is 1-Lipschitz, so physical steering changes remain no
            # larger than the configured per-token increment.
            start_fraction = (start_steering / maximum_steering).clamp(
                -0.999, 0.999
            )
            latent_start = torch.atanh(start_fraction)
            steering = maximum_steering * torch.tanh(
                latent_start
                + torch.cumsum(proposed_delta, dim=-1) / maximum_steering
            )
            curvature = torch.tan(steering) / self.config.vehicle.wheelbase_m
            steering_increment = torch.cat(
                (steering[..., :1] - start_steering, torch.diff(steering, dim=-1)),
                dim=-1,
            )
        else:
            curvature = direct_curvature
            steering = torch.atan(
                self.config.vehicle.wheelbase_m * curvature
            )
            steering_increment = torch.cat(
                (steering[..., :1], torch.diff(steering, dim=-1)), dim=-1
            )
        raw_step_magnitude = (
            1.10 * self.config.data.primitive_step_m
            * torch.sigmoid(self.step(query).squeeze(-1))
        )
        gear_logits = self.gear(query)
        direction_soft = torch.tanh(gear_logits[..., 1] - gear_logits[..., 0])
        # A soft direction is equivalent to a hard forward/reverse gear with
        # a confidence-scaled travel distance.  Expose that *executed* step to
        # the losses and runtime, while retaining the smooth formulation here
        # so gradients still reach the gear head.  Previously training used
        # direction_soft but inference replaced it with +/-1 without scaling
        # the step; on the trained model that single mismatch reduced exact
        # success from roughly 40% to 27% on a 1k-scene diagnostic.
        step_magnitude = raw_step_magnitude * direction_soft.abs()
        poses = integrate_primitives(
            curvature,
            direction_soft,
            self.config.data.primitive_step_m,
            raw_step_magnitude,
        )
        unanchored = {
            "poses": poses,
            "curvature": curvature,
            "steering": steering,
            "steering_increment": steering_increment,
            "step_magnitude": step_magnitude,
            "raw_step_magnitude": raw_step_magnitude,
        }
        length_logits = self.length(query).squeeze(-1)
        training_anchor_lengths = (
            batch.get("target_length") if use_batch_terminal_lengths else None
        )
        if (
            self.config.model.hard_goal_terminal_pose
            and training_anchor_lengths is not None
        ):
            poses = anchor_terminal_pose_trace(
                poses,
                batch["goal"],
                training_anchor_lengths,
                self.config.model.terminal_anchor_steps,
            )
            anchored_controls = controls_from_pose_trace(
                poses,
                direction_soft,
                training_anchor_lengths,
                self.config.vehicle.wheelbase_m,
            )
            curvature = anchored_controls["curvature"]
            steering = anchored_controls["steering"]
            steering_increment = anchored_controls["steering_increment"]
            step_magnitude = anchored_controls["step_magnitude"]
            raw_step_magnitude = anchored_controls["raw_step_magnitude"]
        refined_xy = torch.stack(
            (
                (reference[..., 0] - 0.5) * self.config.data.extent_m
                + self.config.data.raster_center_x_m,
                (0.5 - reference[..., 1]) * self.config.data.extent_m
                + self.config.data.raster_center_y_m,
            ),
            dim=-1,
        )
        output = {
            "poses": poses,
            "curvature": curvature,
            "steering": steering,
            "steering_increment": steering_increment,
            "step_magnitude": step_magnitude,
            "raw_step_magnitude": raw_step_magnitude,
            "gear_logits": gear_logits,
            "length_logits": length_logits,
            # Ranking is a second-stage readout over generated paths. Detach
            # the mature decoder features so the initially random score loss
            # trains the score head without perturbing path generation.
            "scores": self.score(query.mean(dim=2).detach()).squeeze(-1),
            "coarse_pose": coarse_pose,
            "refined_xy": refined_xy,
            "direction_soft": direction_soft,
        }
        if self.config.model.hard_goal_terminal_pose:
            output.update(
                {
                    "unanchored_poses": unanchored["poses"],
                    "unanchored_curvature": unanchored["curvature"],
                    "unanchored_steering": unanchored["steering"],
                    "unanchored_steering_increment": unanchored[
                        "steering_increment"
                    ],
                    "unanchored_step_magnitude": unanchored[
                        "step_magnitude"
                    ],
                    "unanchored_raw_step_magnitude": unanchored[
                        "raw_step_magnitude"
                    ],
                }
            )
        if self.config.data.dense_substeps > 1:
            dense = densify_control_rollout(
                unanchored["steering"],
                direction_soft,
                unanchored["raw_step_magnitude"],
                wheelbase_m=self.config.vehicle.wheelbase_m,
                substeps=self.config.data.dense_substeps,
            )
            raw_dense = dict(dense)
            dense_anchor_lengths = None
            if (
                self.config.model.hard_goal_terminal_pose
                and training_anchor_lengths is not None
            ):
                dense_anchor_lengths = batch.get("dense_target_length")
                if dense_anchor_lengths is None:
                    dense_anchor_lengths = (
                        (training_anchor_lengths - 1)
                        * self.config.data.dense_substeps
                        + 1
                    )
                dense["poses"] = anchor_terminal_pose_trace(
                    dense["poses"],
                    batch["goal"],
                    dense_anchor_lengths,
                    (
                        self.config.model.terminal_anchor_steps
                        * self.config.data.dense_substeps
                    ),
                )
                dense_controls = controls_from_pose_trace(
                    dense["poses"],
                    dense["direction"],
                    dense_anchor_lengths,
                    self.config.vehicle.wheelbase_m,
                )
                dense["steering"] = dense_controls["steering"]
                dense["curvature"] = dense_controls["curvature"]
                dense["step_magnitude"] = dense_controls["step_magnitude"]
                dense["raw_step_magnitude"] = dense_controls[
                    "raw_step_magnitude"
                ]
            output.update({
                "dense_poses": dense["poses"],
                "dense_steering": dense["steering"],
                "dense_curvature": dense["curvature"],
                "dense_direction_soft": dense["direction"],
                "dense_step_magnitude": dense["step_magnitude"],
            })
            if self.config.model.hard_goal_terminal_pose:
                output.update(
                    {
                        "unanchored_dense_poses": raw_dense["poses"],
                        "unanchored_dense_steering": raw_dense["steering"],
                        "unanchored_dense_curvature": raw_dense["curvature"],
                        "unanchored_dense_step_magnitude": raw_dense[
                            "step_magnitude"
                        ],
                    }
                )
        return output

    @torch.no_grad()
    def infer(
        self,
        batch: Mapping[str, torch.Tensor],
        candidate_count: int | None = None,
    ) -> dict[str, torch.Tensor]:
        output = self.forward(
            batch,
            candidate_count=candidate_count,
            use_batch_terminal_lengths=False,
        )
        gear = output["gear_logits"].argmax(dim=-1)
        direction = gear.to(output["curvature"].dtype).mul(2.0).sub(1.0)
        # step_magnitude already contains abs(direction_soft), making this
        # hard-gear integration physically executable and exactly consistent
        # with the differentiable trajectory used during training.
        coarse_runtime_poses_raw = integrate_primitives(
            output["curvature"], direction, self.config.data.primitive_step_m,
            output["step_magnitude"],
        )
        output["gear"] = gear
        coarse_lengths = (output["length_logits"].argmax(dim=-1) + 1).clamp_min(2)
        coarse_runtime_poses = coarse_runtime_poses_raw
        coarse_runtime_curvature = output["curvature"]
        coarse_runtime_steering = output["steering"]
        coarse_runtime_step = output["step_magnitude"]
        if self.config.model.hard_goal_terminal_pose:
            coarse_runtime_poses = anchor_terminal_pose_trace(
                coarse_runtime_poses_raw,
                batch["goal"],
                coarse_lengths,
                self.config.model.terminal_anchor_steps,
            )
            coarse_controls = controls_from_pose_trace(
                coarse_runtime_poses,
                direction,
                coarse_lengths,
                self.config.vehicle.wheelbase_m,
            )
            coarse_runtime_curvature = coarse_controls["curvature"]
            coarse_runtime_steering = coarse_controls["steering"]
            coarse_runtime_step = coarse_controls["step_magnitude"]
            output["unanchored_coarse_runtime_poses"] = coarse_runtime_poses_raw
        if self.config.data.dense_substeps > 1:
            dense = densify_control_rollout(
                output["steering"],
                direction,
                output["step_magnitude"],
                wheelbase_m=self.config.vehicle.wheelbase_m,
                substeps=self.config.data.dense_substeps,
            )
            substeps = self.config.data.dense_substeps
            dense_gear = gear[..., :-1, None].expand(
                *gear.shape[:-1], gear.shape[-1] - 1, substeps
            ).flatten(start_dim=-2)
            dense_gear = torch.cat((dense_gear, gear[..., -1:]), dim=-1)
            dense_lengths = (coarse_lengths - 1) * substeps + 1
            dense_poses_raw = dense["poses"]
            if self.config.model.hard_goal_terminal_pose:
                anchor_position_error, anchor_yaw_error = terminal_pose_error(
                    dense_poses_raw, batch["goal"], dense_lengths
                )
                dense["poses"] = anchor_terminal_pose_trace(
                    dense_poses_raw,
                    batch["goal"],
                    dense_lengths,
                    self.config.model.terminal_anchor_steps * substeps,
                )
                dense_controls = controls_from_pose_trace(
                    dense["poses"],
                    dense["direction"],
                    dense_lengths,
                    self.config.vehicle.wheelbase_m,
                )
                dense["steering"] = dense_controls["steering"]
                dense["curvature"] = dense_controls["curvature"]
                dense["step_magnitude"] = dense_controls["step_magnitude"]
                dense["raw_step_magnitude"] = dense_controls[
                    "raw_step_magnitude"
                ]
                output["unanchored_runtime_poses"] = dense_poses_raw
                output["terminal_anchor_position_correction_m"] = (
                    anchor_position_error
                )
                output["terminal_anchor_yaw_correction_rad"] = anchor_yaw_error
            output.update({
                "coarse_runtime_poses": coarse_runtime_poses,
                "coarse_runtime_steering": coarse_runtime_steering,
                "coarse_runtime_curvature": coarse_runtime_curvature,
                "coarse_runtime_step_magnitude": coarse_runtime_step,
                "coarse_runtime_gear": gear,
                "coarse_runtime_lengths": coarse_lengths,
                "poses": dense["poses"],
                "steering": dense["steering"],
                "steering_increment": torch.cat(
                    (
                        dense["steering"][..., :1],
                        torch.diff(dense["steering"], dim=-1),
                    ),
                    dim=-1,
                ),
                "curvature": dense["curvature"],
                "step_magnitude": dense["step_magnitude"],
                "raw_step_magnitude": dense["raw_step_magnitude"],
                "gear": dense_gear,
                "lengths": dense_lengths,
            })
        else:
            if self.config.model.hard_goal_terminal_pose:
                anchor_position_error, anchor_yaw_error = terminal_pose_error(
                    coarse_runtime_poses_raw, batch["goal"], coarse_lengths
                )
                output["terminal_anchor_position_correction_m"] = (
                    anchor_position_error
                )
                output["terminal_anchor_yaw_correction_rad"] = anchor_yaw_error
            output["poses"] = coarse_runtime_poses
            output["curvature"] = coarse_runtime_curvature
            output["steering"] = coarse_runtime_steering
            output["step_magnitude"] = coarse_runtime_step
            output["raw_step_magnitude"] = coarse_runtime_step
            output["steering_increment"] = torch.cat(
                (
                    coarse_runtime_steering[..., :1],
                    torch.diff(coarse_runtime_steering, dim=-1),
                ),
                dim=-1,
            )
            output["lengths"] = coarse_lengths
        output["selected"] = output["scores"].argmax(dim=-1)
        return output
