"""
BiFPN (Bi-directional Feature Pyramid Network) Implementation for YOLOv8
Based on EfficientDet paper: https://arxiv.org/abs/1911.09070

This module implements a weighted feature fusion mechanism that allows
the model to dynamically prioritize more reliable features, which is
particularly beneficial for underwater images with blur and backscatter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SeparableConv2d(nn.Module):
    """Separable Convolution as used in EfficientDet"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                 stride, padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.swish = nn.SiLU()  # Swish activation (same as SiLU in PyTorch)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.swish(x)


class BiFPNBlock(nn.Module):
    """
    Single BiFPN block with weighted feature fusion.
    
    The key innovation is learnable weights for each input feature map,
    allowing the model to prioritize high-quality features over degraded ones.
    """
    def __init__(self, channels: int, num_levels: int = 5, epsilon: float = 1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.num_levels = num_levels
        
        # Learnable weights for feature fusion
        # Each connection gets its own weight
        self.w1 = nn.Parameter(torch.ones(num_levels - 1))  # Top-down path
        self.w2 = nn.Parameter(torch.ones(num_levels - 1))  # Bottom-up path
        
        # Separable convolutions for each level
        self.convs = nn.ModuleList([
            SeparableConv2d(channels, channels) for _ in range(num_levels)
        ])
        
        # Additional convolutions for intermediate nodes
        self.conv_intermediates = nn.ModuleList([
            SeparableConv2d(channels, channels) for _ in range(num_levels - 2)
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of feature maps from different scales
                     [P3, P4, P5, P6, P7] in increasing receptive field order
        
        Returns:
            List of fused feature maps
        """
        assert len(features) == self.num_levels, f"Expected {self.num_levels} features, got {len(features)}"
        
        # Normalize weights to prevent instability
        w1_normalized = F.relu(self.w1) / (torch.sum(F.relu(self.w1)) + self.epsilon)
        w2_normalized = F.relu(self.w2) / (torch.sum(F.relu(self.w2)) + self.epsilon)
        
        # Top-down path (high-level to low-level)
        # Start from the highest level feature
        td_features = [None] * self.num_levels
        td_features[-1] = features[-1]  # P7 stays the same
        
        for i in range(self.num_levels - 2, -1, -1):  # P6 to P3
            # Upsample higher level feature
            upsampled = F.interpolate(td_features[i + 1], 
                                    size=features[i].shape[2:], 
                                    mode='nearest')
            
            # Weighted fusion
            if i == self.num_levels - 2:  # P6
                td_features[i] = self.convs[i](
                    w1_normalized[i] * features[i] + 
                    (1 - w1_normalized[i]) * upsampled
                )
            else:  # P5, P4, P3
                td_features[i] = self.conv_intermediates[i](
                    w1_normalized[i] * features[i] + 
                    (1 - w1_normalized[i]) * upsampled
                )
        
        # Bottom-up path (low-level to high-level)
        out_features = [None] * self.num_levels
        out_features[0] = td_features[0]  # P3 from top-down
        
        for i in range(1, self.num_levels):  # P4 to P7
            # Downsample lower level feature
            if i == 1:  # P4
                downsampled = F.max_pool2d(out_features[i - 1], kernel_size=2, stride=2)
            else:  # P5, P6, P7
                downsampled = F.max_pool2d(out_features[i - 1], kernel_size=2, stride=2)
            
            # Weighted fusion with original feature and top-down feature
            if i < self.num_levels - 1:
                out_features[i] = self.convs[i](
                    w2_normalized[i - 1] * td_features[i] + 
                    (1 - w2_normalized[i - 1]) * downsampled
                )
            else:  # P7
                out_features[i] = self.convs[i](
                    td_features[i] + downsampled
                )
        
        return out_features


class BiFPNLayer(nn.Module):
    """
    Single BiFPN layer that can be inserted into YOLOv8 architecture.
    Compatible with YOLOv8's module parsing system.
    """
    def __init__(self, c1: int, c2: int = None, epsilon: float = 1e-4):
        super().__init__()
        if c2 is None:
            c2 = c1
        
        self.epsilon = epsilon
        
        # Learnable weights for two-input fusion (most common case)
        self.w1 = nn.Parameter(torch.ones(2))
        self.w2 = nn.Parameter(torch.ones(2))
        
        # Separable convolution for output
        self.conv = SeparableConv2d(c1, c2)
        
    def forward(self, x):
        """
        Forward pass for BiFPN layer.
        Expects input to be either a single tensor or list of tensors.
        """
        if isinstance(x, (list, tuple)):
            if len(x) == 2:
                # Two-input weighted fusion
                w_normalized = F.relu(self.w1) / (torch.sum(F.relu(self.w1)) + self.epsilon)
                fused = w_normalized[0] * x[0] + w_normalized[1] * x[1]
            else:
                # Fallback: simple concatenation for more inputs
                fused = torch.cat(x, dim=1) if len(x) > 1 else x[0]
        else:
            # Single input
            fused = x
            
        return self.conv(fused)


class BiFPN(nn.Module):
    """
    Multi-layer BiFPN for YOLOv8 neck replacement.
    Simplified version that works with YOLOv8's architecture.
    """
    def __init__(self, c1: int, c2: int = None, num_layers: int = 2):
        super().__init__()
        if c2 is None:
            c2 = c1
            
        self.num_layers = num_layers
        
        # Stack of BiFPN layers
        self.layers = nn.ModuleList([
            BiFPNLayer(c1 if i == 0 else c2, c2) for i in range(num_layers)
        ])
        
        # Output projection
        self.out_conv = nn.Conv2d(c2, c2, 1, bias=False)
        self.out_bn = nn.BatchNorm2d(c2)
        self.out_act = nn.SiLU()
    
    def forward(self, x):
        """Forward pass through BiFPN layers"""
        current = x
        for layer in self.layers:
            current = layer(current)
        
        # Final output processing
        out = self.out_conv(current)
        out = self.out_bn(out)
        return self.out_act(out)


# Register the module for YOLOv8 compatibility
def register_bifpn():
    """Register BiFPN modules with ultralytics"""
    try:
        import ultralytics.nn.tasks as tasks
        tasks.BiFPN = BiFPN
        tasks.BiFPNBlock = BiFPNBlock
        tasks.BiFPNLayer = BiFPNLayer
        print("BiFPN modules registered successfully!")
    except ImportError:
        print("Warning: Could not register BiFPN modules with ultralytics")


if __name__ == "__main__":
    # Test the BiFPN implementation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy feature maps (batch_size=2, channels=256)
    p3 = torch.randn(2, 256, 80, 80).to(device)   # High resolution
    p4 = torch.randn(2, 256, 40, 40).to(device)   # Medium resolution  
    p5 = torch.randn(2, 256, 20, 20).to(device)   # Low resolution
    
    features = [p3, p4, p5]
    
    # Test BiFPN
    bifpn = BiFPN(channels=256, num_layers=3, num_levels=3).to(device)
    
    print("Testing BiFPN...")
    print(f"Input shapes: {[f.shape for f in features]}")
    
    with torch.no_grad():
        output_features = bifpn(features)
    
    print(f"Output shapes: {[f.shape for f in output_features]}")
    print("BiFPN test completed successfully!")
    
    # Count parameters
    total_params = sum(p.numel() for p in bifpn.parameters())
    print(f"Total BiFPN parameters: {total_params:,}")