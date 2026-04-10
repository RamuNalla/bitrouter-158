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
        # 1. Scale: Find the average size of the weights
        scale = weight.abs().mean().clamp(min=1e-8)
        
        # 2. Quantize: Divide by scale, round to nearest integer, and clamp between -1 and 1
        weight_scaled = weight / scale
        weight_quantized = torch.clamp(torch.round(weight_scaled), min=-1.0, max=1.0)
        
        # We save the scale for the backward pass if needed, though pure STE doesn't strictly need it
        ctx.save_for_backward(scale)
        
        return weight_quantized

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator:
        # We pretend the forward pass was an identity function for the sake of gradients.
        # The gradients flow straight through the quantization block unchanged.
        return grad_output