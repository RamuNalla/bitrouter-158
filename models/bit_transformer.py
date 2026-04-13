import torch
import torch.nn as nn
from .bit_linear import BitLinear

class BitFeedForward(nn.Module):
    """
    The 1.58-bit Feed Forward Network.
    This is where the massive memory savings occur.
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # Using our custom ternary quantized layers
        self.w1 = BitLinear(dim, hidden_dim)
        self.act = nn.GELU()
        self.w2 = BitLinear(hidden_dim, dim)

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


class BitTransformerBlock(nn.Module):
    """
    A hybrid Transformer block implementing 'Dynamic Bit-Mixing'.
    Attention stays in high precision; FFN is quantized to 1.58-bit.
    """
    def __init__(self, dim: int, num_heads: int, hidden_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        # Standard PyTorch MultiHead Attention (High Precision)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        
        # 1.58-bit Feed Forward Network (Ternary Precision)
        self.ffn = BitFeedForward(dim, hidden_dim)

    def forward(self, x):
        # Pre-LayerNorm architecture (standard for modern LLMs)
        # 1. Attention Block
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        # 2. Feed-Forward Block
        x = x + self.ffn(self.norm2(x))
        return x


class BitRouterModel(nn.Module):
    """
    The complete Micro-LLM used for routing requests.
    """
    def __init__(self, vocab_size: int, dim: int, num_heads: int, hidden_dim: int, num_layers: int, num_classes: int, max_seq_len: int = 256):
        super().__init__()
        # Token and Position Embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # Stack the Hybrid Transformer Blocks
        self.blocks = nn.ModuleList([
            BitTransformerBlock(dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)
        
        # The Final Classification Head (High Precision for final decision)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        seq_len = x.size(1)
        # Generate position IDs (0, 1, 2, ... seq_len-1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)

        # Combine embeddings
        x = self.token_emb(x) + self.pos_emb(positions)

        # Pass through the transformer layers
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Mean pooling: Average the sequence to get a single vector representing the whole prompt
        x_pooled = x.mean(dim=1)

        # Project to the number of routing categories (e.g., Simple vs. Complex)
        logits = self.classifier(x_pooled)
        return logits