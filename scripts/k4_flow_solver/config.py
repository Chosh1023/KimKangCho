from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class SolverConfig:
    intervals: int = 512
    max_phases: int = 6
    context_dim: int = 192
    temporal_channels: int = 48
    path_tokens: int = 256
    max_scene_tokens: int = 2048
    wheelbase: float = 3.004647
    max_speed: float = 2.5
    max_accel: float = 2.0
    max_steer_rate: float = 0.6
    max_steer: float = 0.9809930780518399
    control_margin: float = 2e-4
    clearance: float = 0.1
    latent_epsilon: float = 1e-9
    coordinate_scale: float = 20.0
    max_total_length: float = 80.0
    # Old datasets/checkpoints retain the original strict teacher contract.
    # Runtime/fine-tuning can explicitly choose practical parking tolerances.
    terminal_position_tolerance_m: float = 1e-7
    terminal_yaw_tolerance_rad: float = 1e-7

    def to_dict(self):
        return asdict(self)

    def __post_init__(self):
        if self.intervals < 8 or self.max_phases < 1:
            raise ValueError("invalid control grid or phase count")
        if any(not math.isfinite(v) or v<=0 for v in
               (self.terminal_position_tolerance_m,self.terminal_yaw_tolerance_rad)):
            raise ValueError('goal tolerances must be finite and positive')


DEFAULT = SolverConfig()
