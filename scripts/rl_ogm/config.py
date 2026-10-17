"""Single-source configuration for RL-OGM training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict


@dataclass(frozen=True)
class GridConfig:
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0
    resolution: float = 0.1
    height: int = 200
    width: int = 200
    clearance_clip_m: float = 5.0

    def validate(self) -> None:
        if self.height != 200 or self.width != 200:
            raise ValueError("the live planner contract requires a 200x200 OGM")
        expected_x = (self.x_max - self.x_min) / self.resolution
        expected_y = (self.y_max - self.y_min) / self.resolution
        if abs(expected_x - self.height) > 1e-6 or abs(expected_y - self.width) > 1e-6:
            raise ValueError("grid bounds, resolution, and raster shape disagree")


@dataclass(frozen=True)
class ShieldConfig:
    """HOPE-style action mask followed by exact selected-action validation."""

    # HOPE evaluates 21 steering values in both directions (42 primitives).
    steering_bins: int = 21
    obstacle_ray_bins: int = 120
    obstacle_range_m: float = 10.0
    action_mask_substeps: int = 10
    decision_dt_s: float = 0.50
    integration_step_m: float = 0.05
    forward_speed_mps: float = 1.20
    reverse_speed_mps: float = 1.00
    response_buffer_m: float = 0.08
    minimum_motion_mps: float = 0.04
    steering_rate_deg_s: float = 110.0

    def validate(self) -> None:
        if self.steering_bins < 5 or self.steering_bins % 2 == 0:
            raise ValueError("steering_bins must be an odd integer >= 5")
        if self.decision_dt_s <= 0.0 or self.integration_step_m <= 0.0:
            raise ValueError("shield time and distance steps must be positive")
        if self.obstacle_ray_bins < 32 or self.obstacle_range_m <= 0.0:
            raise ValueError("obstacle ray configuration is invalid")
        if self.action_mask_substeps < 2:
            raise ValueError("action_mask_substeps must be at least two")


@dataclass(frozen=True)
class RewardConfig:
    success: float = 100.0
    collision: float = -100.0
    out_of_bounds: float = -100.0
    timeout: float = -15.0
    step: float = -0.05
    potential_scale: float = 6.0
    # Every procedurally generated training scene carries an exact, collision-
    # free witness.  Progress along that witness is a substantially better
    # shaping potential than Euclidean goal distance for multi-cusp parking:
    # the correct first move is often temporarily away from the stall.
    route_progress_scale: float = 4.0
    route_cross_track_weight: float = 1.25
    route_heading_weight: float = 0.55
    route_direction_mismatch_weight: float = 0.35
    route_projection_backtrack_m: float = 1.25
    route_projection_lookahead_m: float = 3.25
    clearance_scale: float = 0.15
    clearance_threshold_m: float = 0.45
    steer_rate_scale: float = 0.10
    # Necessary witness cusps must not be suppressed.  A small cost still
    # discourages gratuitous oscillation; repeated-state penalties handle the
    # pathological case more directly.
    gear_switch: float = -0.05
    stopped_away_from_goal: float = -0.08
    route_regression_scale: float = 0.50


@dataclass(frozen=True)
class EnvConfig:
    max_decisions: int = 200
    # Zero preserves deployment behavior. The trainer overrides this so the
    # actor must contribute a meaningful prefix before the exact RS suffix.
    connector_min_decisions: int = 0
    goal_xy_tolerance_m: float = 0.20
    goal_yaw_tolerance_deg: float = 4.0
    connector_check_distance_m: float = 10.0
    terminal_reverse_m: float = 1.25
    terminal_reverse_candidates: tuple = (1.25, 1.50, 1.80, 2.10)
    connector_sample_m: float = 0.05
    exact_validation_sample_m: float = 0.05
    shield: ShieldConfig = field(default_factory=ShieldConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass(frozen=True)
class SACConfig:
    raster_channels: int = 1
    vector_dim: int = 16
    action_dim: int = 2
    obstacle_dim: int = 120
    latent_dim: int = 128
    encoder_architecture: str = "hope_multimodal_transformer_v1"
    network_raster_size: int = 64
    transformer_dim: int = 128
    transformer_layers: int = 1
    transformer_heads: int = 8
    transformer_mlp_ratio: float = 1.0
    transformer_dropout: float = 0.0
    transformer_token_grid: int = 1
    batch_size: int = 128
    replay_capacity: int = 150000
    warmup_steps: int = 10240
    gamma: float = 0.995
    tau: float = 0.005
    actor_lr: float = 3.0e-5
    critic_lr: float = 3.0e-5
    encoder_lr_scale: float = 1.0
    alpha_lr: float = 3.0e-5
    initial_alpha: float = 0.10
    # Optional entropy floor.  Zero preserves the original SAC behaviour.
    # Dynamic avoidance uses a non-zero value so exploration cannot collapse
    # after a short run of easy guide-following episodes.
    min_alpha: float = 0.0
    # Let a newly initialized Critic fit the current reward/transition
    # contract before it is allowed to move a warm-started Actor.  This is
    # especially important when only the Actor is transferred from an older
    # controller whose reward function or safety shield was different.
    critic_warmup_updates: int = 0
    weight_decay: float = 0.0
    target_entropy: float = -2.0
    # The public HOPE SAC implementation performs one update per ten samples.
    update_every: int = 10
    updates_per_step: int = 1
    policy_delay: int = 2
    grad_clip: float = 10.0
    # Modest PER keeps rare success/cusp transitions visible without allowing
    # one large terminal TD error to dominate a whole update.
    per_alpha: float = 0.40
    per_beta_start: float = 0.40
    per_beta_steps: int = 500000
    n_step: int = 5
    # Training-only route demonstrations.  They never enter the deployment
    # observation or planner contract.
    expert_capacity: int = 50000
    expert_batch_size: int = 64
    expert_min_samples: int = 128
    expert_samples_per_scenario: int = 8
    expert_bc_initial_weight: float = 4.0
    expert_bc_final_weight: float = 0.25
    expert_bc_anneal_steps: int = 300000
    expert_speed_loss_weight: float = 2.0
    expert_pretrain_updates_per_scenario: int = 4
    amp: bool = True
    amp_dtype: str = "bfloat16"
    drq_translation_augmentation: bool = False
    compile_model: bool = False
    compile_mode: str = "default"

    def transformer_kwargs(self) -> Dict[str, object]:
        if self.encoder_architecture != "hope_multimodal_transformer_v1":
            raise ValueError("unsupported RL-OGM encoder: {}".format(self.encoder_architecture))
        return {
            "obstacle_dim": int(self.obstacle_dim),
            "network_raster_size": int(self.network_raster_size),
            "transformer_dim": int(self.transformer_dim),
            "transformer_layers": int(self.transformer_layers),
            "transformer_heads": int(self.transformer_heads),
            "transformer_mlp_ratio": float(self.transformer_mlp_ratio),
            "transformer_dropout": float(self.transformer_dropout),
            "transformer_token_grid": int(self.transformer_token_grid),
        }


def config_dict(config) -> Dict[str, object]:
    return asdict(config)
