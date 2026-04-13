import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.bit_transformer import BitRouterModel

def test_model_forward_pass():
    """Verify that the model can process integer tokens and output classification logits."""
    # Hyperparameters for a tiny "Micro-Model"
    vocab_size = 1000
    dim = 128
    num_heads = 4
    hidden_dim = 512
    num_layers = 2
    num_classes = 2 # e.g., 0 = Stay Local, 1 = Route to Cloud API
    max_seq_len = 64
    batch_size = 4

    # Initialize model
    model = BitRouterModel(
        vocab_size=vocab_size,
        dim=dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        max_seq_len=max_seq_len
    )

    # Create dummy token data (like text converted to IDs)
    dummy_input = torch.randint(0, vocab_size, (batch_size, max_seq_len))

    # Forward pass
    output = model(dummy_input)

    # Check output shape: Should be [Batch Size, Number of Classes]
    assert output.shape == (batch_size, num_classes), f"Expected shape {(batch_size, num_classes)}, got {output.shape}"
    
    print("✅ Transformer forward pass successful.")
    print(f"Logits shape: {output.shape}")

if __name__ == "__main__":
    test_model_forward_pass()