"""Configuration contract for the HOPE path Transformer pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class VehicleConfig:
    wheelbase_m: float = 3.004647
    front_overhang_m: float = 0.80687972
    rear_overhang_m: float = 0.98025278
    width_m: float = 2.16345
    max_steer_rad: float = 0.980994

    @property
    def max_curvature_m_inv(self) -> float:
        import math

        return math.tan(self.max_steer_rad) / self.wheelbase_m


@dataclass(frozen=True)
class DataConfig:
    extent_m: float = 40.0
    raster_size: int = 256
    raster_center_x_m: float = 0.0
    raster_center_y_m: float = 0.0
    sdf_clip_m: float = 3.0
    primitive_step_m: float = 0.20
    primitive_count: int = 192
    # Number of controller-facing micro-steps represented by one Transformer
    # motion interval.  A value of four keeps 192 attention tokens while
    # supervising/emitting an approximately 0.05 m path.
    dense_substeps: int = 1
    coarse_count: int = 48
    max_obstacles: int = 16
    max_vertices: int = 8
    max_gear_switches: int = 12
    goal_position_tolerance_m: float = 0.30
    goal_yaw_tolerance_deg: float = 5.0
    minimum_clearance_m: float = 0.0
    level_names: tuple[str, ...] = ("Normal", "Complex", "Extrem")
    # A geometrically valid path can still be impossible for the low-level
    # controller to track.  When enabled, validation/runtime selection also
    # enforces these steering-envelope limits.
    trackability_gate_enabled: bool = False
    trackability_max_steering_variation_deg_per_m: float = 40.0
    trackability_max_same_gear_steer_jump_deg: float = 20.0
    trackability_max_terminal_steer_deg: float = 10.0
    trackability_max_saturation_fraction: float = 0.45

    @property
    def resolution_m(self) -> float:
        return self.extent_m / float(self.raster_size)

    @property
    def dense_step_m(self) -> float:
        return self.primitive_step_m / float(self.dense_substeps)

    @property
    def dense_primitive_count(self) -> int:
        return (self.primitive_count - 1) * self.dense_substeps + 1


@dataclass(frozen=True)
class ModelConfig:
    raster_channels: int = 5
    hidden_dim: int = 256
    attention_heads: int = 8
    vector_layers: int = 3
    coarse_layers: int = 6
    refinement_layers: int = 3
    deformable_points: int = 4
    candidate_count: int = 4
    dropout: float = 0.10
    backbone: str = "convnext_tiny"
    condition_start_direction: bool = False
    bounded_steering_increments: bool = False
    max_steering_delta_deg: float = 18.0
    # Keep the learned prefix, but distribute its terminal SE(2) error over
    # the last few motion tokens so the executable path ends at the exact
    # rear-axle goal pose. Legacy checkpoints leave this disabled.
    hard_goal_terminal_pose: bool = False
    terminal_anchor_steps: int = 24
    terminal_anchor_max_position_correction_m: float = 0.75
    terminal_anchor_max_yaw_correction_deg: float = 10.0


@dataclass(frozen=True)
class LossConfig:
    position: float = 1.0
    yaw: float = 0.5
    curvature: float = 0.5
    step: float = 0.5
    gear: float = 0.5
    stop: float = 0.2
    goal: float = 2.0
    collision: float = 3.0
    smoothness: float = 0.2
    # All-candidate trackability objectives.  Defaults remain zero so legacy
    # configurations/checkpoints retain their original behavior.
    steering_rate: float = 0.0
    steering_jerk: float = 0.0
    terminal_steering: float = 0.0
    steering_saturation: float = 0.0
    gear_flicker: float = 0.0
    ranking_trackability: float = 0.0
    goal_tolerance_hinge: bool = False
    dense_position: float = 0.0
    dense_yaw: float = 0.0
    dense_steering: float = 0.0
    diversity: float = 0.1
    ranking: float = 0.5
    # Oracle-style hinge on the runtime goal envelope. Unlike the continuous
    # goal term, it is exactly zero once a candidate is within 0.30 m / 5 deg.
    feasibility: float = 0.0
    collision_margin_m: float = 0.10
    # Collision supervision mirrors the runtime's swept-polygon gate more
    # closely than a uniform mean over sparse path nodes.
    collision_topk_points: int = 3
    collision_topk_steps: int = 8
    collision_mean_weight: float = 0.25
    predicted_endpoint_goal_weight: float = 0.50
    # Blend anonymous Best-of-K matching (0) toward deterministic eight-way
    # topology assignment (1).  The topology-conditioned recipe uses one
    # query for each initial-gear/switch-complexity/detour-side family.
    topology_assignment_weight: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 40
    base_epochs: int = 10
    ranking_start_epoch: int = 30
    batch_size: int = 32
    gradient_accumulation: int = 1
    workers: int = 4
    validation_interval: int = 5
    learning_rate: float = 3.0e-4
    backbone_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    warmup_fraction: float = 0.05
    gradient_clip: float = 1.0
    amp: bool = True
    ema_decay: float = 0.999
    early_stop_patience: int = 6
    extreme_sample_weight: float = 2.0
    # Equalize effective forward-first and reverse-first GT exposure from the
    # actual saved path, independent of the collector's requested search hint.
    balance_initial_direction: bool = False
    # Equalize the effective sampling mass of the eight GT topology families.
    # This is stronger than direction balancing and should not be enabled at
    # the same time.
    balance_topology_modes: bool = False


@dataclass(frozen=True)
class HopePathConfig:
    schema_version: str = "hope-path-transformer-config-v1"
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        if self.data.raster_size < 64 or self.data.raster_size % 16:
            raise ValueError("raster_size must be >=64 and divisible by 16")
        if self.data.primitive_count < self.data.coarse_count:
            raise ValueError("primitive_count must be >= coarse_count")
        if self.model.hidden_dim % self.model.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if self.model.candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        if self.model.max_steering_delta_deg <= 0.0:
            raise ValueError("max_steering_delta_deg must be positive")
        if self.model.terminal_anchor_steps < 2:
            raise ValueError("terminal_anchor_steps must be at least two")
        if (
            self.model.terminal_anchor_max_position_correction_m <= 0.0
            or self.model.terminal_anchor_max_yaw_correction_deg <= 0.0
        ):
            raise ValueError("terminal anchor correction limits must be positive")
        if self.data.extent_m <= 0.0 or self.data.primitive_step_m <= 0.0:
            raise ValueError("metric grid and primitive steps must be positive")
        if self.data.dense_substeps < 1:
            raise ValueError("dense_substeps must be positive")
        if (
            abs(self.data.raster_center_x_m) >= 0.5 * self.data.extent_m
            or abs(self.data.raster_center_y_m) >= 0.5 * self.data.extent_m
        ):
            raise ValueError("raster center offset must remain inside the raster extent")
        if not (
            0 < self.training.base_epochs
            < self.training.ranking_start_epoch
            < self.training.epochs
        ):
            raise ValueError(
                "training stages must satisfy 0 < base_epochs < "
                "ranking_start_epoch < epochs"
            )
        if self.training.batch_size < 1 or self.training.gradient_accumulation < 1:
            raise ValueError("batch size and gradient accumulation must be positive")
        if self.training.validation_interval < 1:
            raise ValueError("validation_interval must be positive")
        if self.training.early_stop_patience < 1:
            raise ValueError("early_stop_patience must be positive")
        if not 1 <= self.loss.collision_topk_points <= 9:
            raise ValueError("collision_topk_points must be in [1, 9]")
        if self.loss.collision_topk_steps < 1:
            raise ValueError("collision_topk_steps must be positive")
        if not 0.0 <= self.loss.collision_mean_weight <= 1.0:
            raise ValueError("collision_mean_weight must be in [0, 1]")
        if self.loss.predicted_endpoint_goal_weight < 0.0:
            raise ValueError("predicted_endpoint_goal_weight must be non-negative")
        if not 0.0 <= self.loss.topology_assignment_weight <= 1.0:
            raise ValueError("topology_assignment_weight must be in [0, 1]")
        if (
            self.loss.topology_assignment_weight > 0.0
            and self.model.candidate_count != 8
        ):
            raise ValueError(
                "topology-conditioned assignment requires candidate_count=8"
            )
        if (
            self.training.balance_initial_direction
            and self.training.balance_topology_modes
        ):
            raise ValueError(
                "direction and topology balancing cannot both be enabled"
            )
        if self.loss.feasibility < 0.0:
            raise ValueError("feasibility must be non-negative")
        if any(
            value < 0.0
            for value in (
                self.loss.steering_rate,
                self.loss.steering_jerk,
                self.loss.terminal_steering,
                self.loss.steering_saturation,
                self.loss.gear_flicker,
                self.loss.ranking_trackability,
                self.loss.dense_position,
                self.loss.dense_yaw,
                self.loss.dense_steering,
            )
        ):
            raise ValueError("trackability loss weights must be non-negative")
        if (
            self.data.trackability_max_steering_variation_deg_per_m <= 0.0
            or self.data.trackability_max_same_gear_steer_jump_deg <= 0.0
            or self.data.trackability_max_terminal_steer_deg <= 0.0
        ):
            raise ValueError("trackability steering thresholds must be positive")
        if not 0.0 <= self.data.trackability_max_saturation_fraction <= 1.0:
            raise ValueError("trackability_max_saturation_fraction must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def config_from_dict(payload: Dict[str, Any]) -> HopePathConfig:
    config = HopePathConfig(
        schema_version=str(payload.get("schema_version", HopePathConfig.schema_version)),
        vehicle=VehicleConfig(**payload.get("vehicle", {})),
        data=DataConfig(**payload.get("data", {})),
        model=ModelConfig(**payload.get("model", {})),
        loss=LossConfig(**payload.get("loss", {})),
        training=TrainingConfig(**payload.get("training", {})),
    )
    config.validate()
    return config


def load_config(path: str | Path | None = None) -> HopePathConfig:
    if path is None:
        config = HopePathConfig()
        config.validate()
        return config
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return config_from_dict(payload)


def save_config(config: HopePathConfig, path: str | Path) -> None:
    config.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(destination)
