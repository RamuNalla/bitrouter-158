import os
import time
import psutil
import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.bit_transformer import BitRouterModel

# ==========================================
# 1. THE CONTROL GROUP (STANDARD MODEL)
# ==========================================
class StandardFeedForward(nn.Module):
    """Standard PyTorch FFN using heavy Float32 precision."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.w2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))

class StandardTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, hidden_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = StandardFeedForward(dim, hidden_dim)

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

class StandardRouterModel(nn.Module):
    """A 1:1 replica of your BitRouter, but using standard nn.Linear."""
    def __init__(self, vocab_size, dim, num_heads, hidden_dim, num_layers, num_classes, max_seq_len=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([
            StandardTransformerBlock(dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x_pooled = x.mean(dim=1)
        return self.classifier(x_pooled)

# ==========================================
# 2. BENCHMARKING FUNCTIONS
# ==========================================
def measure_latency(model, dummy_input, num_runs=50):
    """Measures the average milliseconds per token during inference."""
    model.eval()
    
    # Warm-up runs (gets the CPU caches ready)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Actual timed runs
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    end_time = time.perf_counter()

    total_time_ms = (end_time - start_time) * 1000
    avg_run_ms = total_time_ms / num_runs
    
    # Calculate per-token latency (Batch Size * Seq Len)
    total_tokens = dummy_input.size(0) * dummy_input.size(1)
    ms_per_token = avg_run_ms / total_tokens
    
    return ms_per_token

def measure_memory():
    """Captures the current process RAM usage in Megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# ==========================================
# 3. THE EXECUTION SUITE
# ==========================================
def run_benchmark():
    print("🚀 Initializing CPU Benchmarking Suite...\n")
    
    # Model Hyperparameters 
    vocab_size = 30522 # bert-tiny vocab size
    dim = 256
    num_heads = 4
    hidden_dim = 1024
    num_layers = 4
    num_classes = 2
    max_seq_len = 64
    batch_size = 1

    dummy_input = torch.randint(0, vocab_size, (batch_size, max_seq_len))

    # --- Benchmark Standard Model ---
    mem_before_std = measure_memory()
    standard_model = StandardRouterModel(vocab_size, dim, num_heads, hidden_dim, num_layers, num_classes)
    mem_after_std = measure_memory()
    std_model_size_mb = mem_after_std - mem_before_std
    
    print("Testing Standard Model (Float32)...")
    std_latency = measure_latency(standard_model, dummy_input)
    
    # Clear memory
    del standard_model
    import gc; gc.collect()

    # --- Benchmark BitRouter Model ---
    mem_before_bit = measure_memory()
    bit_model = BitRouterModel(vocab_size, dim, num_heads, hidden_dim, num_layers, num_classes)
    mem_after_bit = measure_memory()
    bit_model_size_mb = mem_after_bit - mem_before_bit

    print("Testing BitRouter Model (1.58-bit)...")
    bit_latency = measure_latency(bit_model, dummy_input)

    # --- Print Results ---
    print("\n" + "="*50)
    print(f"{'Metric':<25} | {'Standard FP32':<12} | {'BitRouter 1.58b'}")
    print("="*50)
    print(f"{'RAM Footprint (MB)':<25} | {std_model_size_mb:<12.2f} | {bit_model_size_mb:.2f}")
    print(f"{'Latency (ms / token)':<25} | {std_latency:<12.3f} | {bit_latency:.3f}")
    print("="*50)
    
    speedup = std_latency / bit_latency
    ram_reduction = ((std_model_size_mb - bit_model_size_mb) / std_model_size_mb) * 100
    print(f"\n🏆 RESULTS: BitRouter is {speedup:.2f}x faster and uses {ram_reduction:.1f}% less RAM!")

if __name__ == "__main__":
    run_benchmark()