import torch
import torch.nn as nn
import torch.nn.functional as F

class TernaryQuantizeSTE(torch.autograd.Function):
    """
    Custom autograd function to implement the Straight-Through Estimator (STE).
    This allows gradients to flow through the non-differentiable rounding operation
    during the backward pass.
    """
    @staticmethod
    def forward(ctx, weight):
        # 1. Calculate the scale factor (gamma) as the mean absolute value of the weights
        # We add a small epsilon to prevent division by zero
        scale = weight.abs().mean().clamp(min=1e-8)
        
        # 2. Scale, round, and clamp to get values in {-1, 0, 1}
        weight_scaled = weight / scale
        weight_quantized = torch.clamp(torch.round(weight_scaled), min=-1.0, max=1.0)
        
        # We save the scale for the backward pass if needed, though pure STE doesn't strictly need it
        ctx.save_for_backward(scale)
        
        return weight_quantized