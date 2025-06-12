# File: models/eca.py
import torch
import torch.nn as nn
import math

class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA) module.
    This module uses a 1D convolution to efficiently capture local
    cross-channel interactions.
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # Calculate the adaptive kernel size for the 1D convolution
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze spatial dimensions and apply average pooling
        y = self.avg_pool(x)
        
        # Apply the 1D convolution across the channel dimension
        # Reshape for conv1d: (B, C, 1, 1) -> (B, 1, C)
        # Reshape back: (B, 1, C) -> (B, C, 1, 1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        # Apply sigmoid to get attention weights
        y = self.sigmoid(y)
        
        # Apply attention weights to the input feature map
        return x * y.expand_as(x)