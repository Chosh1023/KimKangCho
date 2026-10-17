"""Stochastic policy that emits a compact residual steering spline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for value in (16, 8, 4, 2):
        if channels % value == 0:
            return value
    return 1


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = F.silu(self.norm1(self.conv1(value)), inplace=True)
        value = self.norm2(self.conv2(value))
        return F.silu(value + residual, inplace=True)


class PathRefinerPolicy(nn.Module):
    """Gaussian contextual policy over low-dimensional steering corrections.

    The mean head starts at zero, so deterministic inference initially returns
    the frozen Dense-v4 path exactly. Exploration is isolated in ``log_std``.
    """

    def __init__(
        self,
        raster_channels: int = 4,
        vector_dim: int = 32,
        action_knots: int = 12,
        hidden_dim: int = 256,
        topology_count: int = 1,
    ):
        super().__init__()
        self.topology_count = max(1, int(topology_count))
        channels = (32, 64, 96, 128)
        layers = []
        incoming = int(raster_channels)
        for outgoing in channels:
            layers.extend(
                (
                    nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(_group_count(outgoing), outgoing),
                    nn.SiLU(inplace=True),
                    _ResidualBlock(outgoing),
                )
            )
            incoming = outgoing
        self.raster_encoder = nn.Sequential(*layers)
        self.vector_encoder = nn.Sequential(
            nn.Linear(int(vector_dim), 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 128),
            nn.SiLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            nn.Linear(channels[-1] + 128, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(inplace=True),
        )
        self.topology_embedding = (
            nn.Embedding(self.topology_count, int(hidden_dim))
            if self.topology_count > 1
            else None
        )
        self.mean = nn.Linear(int(hidden_dim), int(action_knots))
        self.log_std = nn.Linear(int(hidden_dim), int(action_knots))
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_std.weight)
        # Exploration starts as a local detour (roughly 4 degrees at one
        # standard deviation with the default 18-degree action scale), not a
        # wholesale replacement of an already useful Dense-v4 path.
        nn.init.constant_(self.log_std.bias, -1.50)

    def _feature(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        raster = observation["raster"].float()
        vector = observation["vector"].float()
        encoded_raster = self.raster_encoder(raster).mean(dim=(-1, -2))
        encoded_vector = self.vector_encoder(vector)
        return self.trunk(torch.cat((encoded_raster, encoded_vector), dim=-1))

    def _topology_ids(
        self,
        batch: int,
        count: int,
        device: torch.device,
    ) -> torch.Tensor:
        ids = torch.arange(int(count), device=device, dtype=torch.long)
        ids = torch.remainder(ids, self.topology_count)
        return ids[None].expand(int(batch), -1)

    def distribution(
        self,
        observation: dict[str, torch.Tensor],
        topology_ids: torch.Tensor | None = None,
    ):
        feature = self._feature(observation)
        if topology_ids is not None:
            if self.topology_embedding is None:
                raise ValueError("topology_ids require topology_count > 1")
            ids = topology_ids.to(device=feature.device, dtype=torch.long)
            if ids.ndim == 1:
                ids = ids[None].expand(len(feature), -1)
            if ids.ndim != 2 or ids.shape[0] != len(feature):
                raise ValueError("topology_ids must have shape [G] or [B,G]")
            feature = feature[:, None] + self.topology_embedding(ids)
        mean = self.mean(feature)
        log_std = self.log_std(feature).clamp(-3.5, 0.25)
        return torch.distributions.Normal(mean, log_std.exp())

    def group_distribution(
        self, observation: dict[str, torch.Tensor], group_size: int
    ):
        if self.topology_embedding is None:
            distribution = self.distribution(observation)
            mean = distribution.mean[:, None].expand(-1, int(group_size), -1)
            scale = distribution.scale[:, None].expand_as(mean)
            return torch.distributions.Normal(mean, scale)
        ids = self._topology_ids(
            len(observation["raster"]),
            int(group_size),
            observation["raster"].device,
        )
        return self.distribution(observation, ids)

    def sample_group(
        self, observation: dict[str, torch.Tensor], group_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.topology_embedding is None:
            distribution = self.distribution(observation)
            latent = distribution.sample((int(group_size),)).permute(1, 0, 2)
            log_probability = distribution.log_prob(
                latent.permute(1, 0, 2)
            ).sum(-1).permute(1, 0)
            entropy = distribution.entropy().sum(dim=-1)
        else:
            distribution = self.group_distribution(observation, group_size)
            latent = distribution.sample()
            log_probability = distribution.log_prob(latent).sum(dim=-1)
            entropy = distribution.entropy().sum(dim=-1)
        action = torch.tanh(latent)
        return action, log_probability, entropy

    def deterministic(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.tanh(self.distribution(observation).mean)

    def deterministic_group(
        self, observation: dict[str, torch.Tensor], candidate_count: int
    ) -> torch.Tensor:
        return torch.tanh(
            self.group_distribution(observation, candidate_count).mean
        )

    def fixed_candidate_group(
        self,
        observation: dict[str, torch.Tensor],
        candidate_count: int,
        *,
        seed: int = 20260826,
    ) -> torch.Tensor:
        """Return a reproducible mean-plus-antithetic candidate set.

        Candidate zero is always the policy mean.  The remaining actions use
        paired Gaussian offsets, so live inference can exploit the policy's
        learned uncertainty without making repeated runs random.  In topology
        mode the first occurrence of every family is its deterministic mean;
        later occurrences are distinct antithetic proposals for that family.
        """

        count = max(1, int(candidate_count))
        if self.topology_embedding is not None:
            distribution = self.group_distribution(observation, count)
            mean = distribution.mean
            if count <= self.topology_count:
                return torch.tanh(mean)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
            noise = torch.zeros(
                (count, mean.shape[-1]), dtype=torch.float32, device="cpu"
            )
            for topology_id in range(self.topology_count):
                family_indices = list(range(topology_id, count, self.topology_count))
                extra = max(0, len(family_indices) - 1)
                if not extra:
                    continue
                pair_count = (extra + 1) // 2
                positive = torch.randn(
                    (pair_count, mean.shape[-1]),
                    generator=generator,
                    dtype=torch.float32,
                    device="cpu",
                ).clamp(-2.0, 2.0)
                paired = torch.stack((positive, -positive), dim=1).reshape(
                    -1, mean.shape[-1]
                )[:extra]
                noise[family_indices[1:]] = paired
            noise = noise.to(device=mean.device, dtype=mean.dtype)
            latent = mean + distribution.stddev * noise[None]
            return torch.tanh(latent)
        distribution = self.distribution(observation)
        mean = distribution.mean
        if count == 1:
            return torch.tanh(mean)[:, None]
        required_noise = count - 1
        pair_count = (required_noise + 1) // 2
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        positive = torch.randn(
            (pair_count, mean.shape[-1]),
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        ).clamp(-2.0, 2.0)
        noise = torch.cat((positive, -positive), dim=0)[:required_noise]
        noise = noise.to(device=mean.device, dtype=mean.dtype)
        latent = mean[:, None] + distribution.stddev[:, None] * noise[None]
        return torch.cat((torch.tanh(mean)[:, None], torch.tanh(latent)), dim=1)
