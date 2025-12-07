"""
Minimal CBAM implementation - just the essentials.
"""
import torch
import torch.nn as nn


class CBAM(nn.Module):
    """Simplest CBAM: channel attention + spatial attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention: squeeze with avg pool, excite with 2-layer MLP
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Spatial attention: 7x7 conv on channel-pooled features
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        x = x * self.fc(self.avg_pool(x))

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_out, max_out], dim=1)
        x = x * self.spatial(spatial_in)

        return x
