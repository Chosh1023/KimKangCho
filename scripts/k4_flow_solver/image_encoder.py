"""Compact ConvNeXt image tokens, with metric positions in the solver frame."""
import torch
from torch import nn
from torch.nn import functional as F


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.expand = nn.Linear(channels, channels*4)
        self.project = nn.Linear(channels*4, channels)
        self.scale = nn.Parameter(torch.full((channels,), 1e-6))

    def forward(self, x):
        y = self.depthwise(x).permute(0, 2, 3, 1)
        y = self.project(F.gelu(self.expand(self.norm(y))))*self.scale
        return x+y.permute(0, 3, 1, 2)


class SpatialImageEncoder(nn.Module):
    def __init__(self, dimension, image_config, coordinate_scale=20.):
        super().__init__()
        self.config = dict(image_config)
        self.coordinate_scale = coordinate_scale
        channels = len(image_config['channels'])
        self.stem = nn.Sequential(nn.Conv2d(channels, 32, 4, stride=4), nn.GroupNorm(1, 32), ConvNeXtBlock(32))
        self.medium = nn.Sequential(nn.Conv2d(32, 64, 2, stride=2), nn.GroupNorm(1, 64), ConvNeXtBlock(64))
        self.coarse = nn.Sequential(nn.AvgPool2d(2, ceil_mode=True), nn.Conv2d(64, 128, 1),
                                    nn.GroupNorm(1, 128), ConvNeXtBlock(128), ConvNeXtBlock(128))
        self.projections = nn.ModuleList([nn.Conv2d(64, dimension, 1), nn.Conv2d(128, dimension, 1)])
        self.position = nn.Sequential(nn.Linear(2, dimension), nn.SiLU(), nn.Linear(dimension, dimension))
        self.scale_kind = nn.Parameter(torch.randn(2, dimension)*.02)
        self.norm = nn.LayerNorm(dimension)

    def metric_positions(self, size, start, dtype):
        xmin, xmax, ymin, ymax = self.config['bounds_m']
        x = xmax-(torch.arange(size, device=start.device, dtype=dtype)+.5)*(xmax-xmin)/size
        y = ymin+(torch.arange(size, device=start.device, dtype=dtype)+.5)*(ymax-ymin)/size
        x, y = torch.meshgrid(x, y, indexing='ij')
        dx, dy = x.flatten()[None]-start[:, 0, None], y.flatten()[None]-start[:, 1, None]
        c, s = start[:, 2, None].cos(), start[:, 2, None].sin()
        return torch.stack((c*dx+s*dy, -s*dx+c*dy), -1)/self.coordinate_scale

    def forward(self, image, start):
        if image.ndim != 4 or image.shape[1:] != (len(self.config['channels']), *self.config['shape']):
            raise ValueError('BEV image channel/grid contract mismatch')
        valid = image[:, self.config['valid_channel']:self.config['valid_channel']+1] > .5
        middle = self.medium(self.stem(image*valid))
        coarse = self.coarse(middle)
        tokens, masks = [], []
        for i, (feature, size) in enumerate(zip((middle, coarse), (8, 4))):
            value = self.projections[i](F.adaptive_avg_pool2d(feature, (size, size))).flatten(2).transpose(1, 2)
            position = self.metric_positions(size, start, image.dtype)
            tokens.append(self.norm(value+self.position(position)+self.scale_kind[i]))
            masks.append(F.adaptive_max_pool2d(valid.float(), (size, size)).flatten(1).bool())
        return torch.cat(tokens, 1), torch.cat(masks, 1)
