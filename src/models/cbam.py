"""
Convolutional Block Attention Module (CBAM) implementation for YOLO.
Paper: "CBAM: Convolutional Block Attention Module" (https://arxiv.org/abs/1807.06521)
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention module using both average and max pooling."""

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """Spatial attention module using channel-wise pooling."""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Args:
        channels (int): Number of input channels (will be auto-adjusted based on actual input)
        reduction (int): Channel reduction ratio for channel attention
        kernel_size (int): Kernel size for spatial attention
    """

    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        # Store parameters for lazy initialization
        self.channels = channels
        self.reduction = reduction
        self.kernel_size = kernel_size
        self.channel_attention = None
        self.spatial_attention = None
        self._initialized = False

    def forward(self, x):
        # Lazy initialization based on actual input channels
        if not self._initialized:
            actual_channels = x.shape[1]  # Get actual number of channels from input
            self.channel_attention = ChannelAttention(actual_channels, self.reduction)
            self.spatial_attention = SpatialAttention(self.kernel_size)
            # Move to same device as input
            self.channel_attention = self.channel_attention.to(x.device)
            self.spatial_attention = self.spatial_attention.to(x.device)
            self._initialized = True

        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
