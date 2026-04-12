import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.bit_linear import BitLinear, TernaryQuantizeSTE

def test_ternary_quantization():
    """Verify that the quantization function strictly outputs -1, 0, or 1."""
    # Create a random weight tensor
    dummy_weights = torch.randn(10, 10) * 10 
    
    # Apply quantization
    quantized_weights = TernaryQuantizeSTE.apply(dummy_weights)
    
    # Check unique values
    unique_vals = torch.unique(quantized_weights)
    
    print(f"Original weights sample: \n{dummy_weights[0, :5]}")
    print(f"Quantized weights sample: \n{quantized_weights[0, :5]}")
    print(f"Unique values in quantized tensor: {unique_vals.tolist()}")
    
    # Assert that all values are within the set {-1.0, 0.0, 1.0}
    assert torch.all(torch.isin(unique_vals, torch.tensor([-1.0, 0.0, 1.0]))), "Weights are not strictly ternary!"
    print("✅ Quantization test passed.")

def test_ste_gradients():
    """Verify that gradients flow back to the latent weights."""
    batch_size, in_dim, out_dim = 4, 16, 8
    
    layer = BitLinear(in_features=in_dim, out_features=out_dim, bias=False)
    x = torch.randn(batch_size, in_dim)
    
    # Forward pass
    output = layer(x)
    
    # Create a dummy loss (e.g., sum of all outputs)
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check if gradients exist and are not zero
    assert layer.weight.grad is not None, "Gradients are None!"
    assert torch.any(layer.weight.grad != 0), "Gradients are exactly zero, STE failed!"
    
    print("✅ Gradient flow (STE) test passed.")

if __name__ == "__main__":
    print("Running Day 1 Unit Tests...\n")
    test_ternary_quantization()
    print("-" * 30)
    test_ste_gradients()
    print("\nAll Day 1 checks complete. Ready for model assembly.")