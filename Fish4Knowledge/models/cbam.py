# File: models/cbam.py
import torch
import torch.nn as nn

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Sequentially applies channel and spatial attention to a feature map.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention: squeeze-and-excite
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                            # (B, C, 1, 1)
            nn.Conv2d(channels, channels // reduction, 1),       # reduce channels
            nn.ReLU(inplace=True),                              # non-linearity
            nn.Conv2d(channels // reduction, channels, 1),       # restore channels
            nn.Sigmoid()                                        # scale 0-1
        )
        # Spatial attention: conv on concatenated statistics
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),  # input: avg+max maps
            nn.Sigmoid()
        )

    def forward(self, x):
        # ----- Channel Attention -----
        ca = self.channel_attention(x)  # (B, C, H, W)
        x = x * ca                      # scale channel-wise

        # ----- Spatial Attention -----
        # compute average and max along channel axis
        avg_out = torch.mean(x, dim=1, keepdim=True)           # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)         # (B, 1, H, W)
        sa_input = torch.cat([avg_out, max_out], dim=1)        # (B, 2, H, W)
        sa = self.spatial_attention(sa_input)                  # (B, 1, H, W)
        x = x * sa                                             # scale spatially
        return x














