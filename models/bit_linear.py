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

class BitLinear(nn.Module):
    """
    A Linear layer where weights are quantized to 1.58 bits {-1, 0, 1} during the forward pass.
    Inspired by the BitNet b1.58 architecture.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super(BitLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # We store the weights in full precision (FP32/FP16) as latent weights.
        # These are the weights that will actually be updated by the optimizer.
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / in_features)**0.5)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Apply LayerNorm as recommended in the BitNet paper before the linear projection
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False)

    def forward(self, x):
        # 1. Normalize the input activations
        x_norm = self.norm(x)
        
        # 2. Quantize the weights to {-1, 0, 1} using our STE function
        # Note: We do this dynamically during every forward pass in training.
        quantized_weight = TernaryQuantizeSTE.apply(self.weight)
        
        # 3. Perform the linear projection using the ternary weights
        out = F.linear(x_norm, quantized_weight, self.bias)
        
        return out

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'