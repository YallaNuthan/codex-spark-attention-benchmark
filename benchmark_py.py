import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

class StandardAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        # Standard implementation: Creates large intermediate tensors (High Memory Usage)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class CodexOptimizedAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # ⚡️ OPTIMIZATION: Fused Kernel (FlashAttention)
        # Codex correctly identified this API as the fix for OOM errors.
        x = F.scaled_dot_product_attention(q, k, v)
        
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

def benchmark(model, x, runs=50):
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) / runs * 1000 # Convert to ms

if __name__ == "__main__":
    # Configuration
    DIM = 512
    HEADS = 8
    SEQ_LEN = 2048 # Adjusted to 2048 to fit T4 GPU Memory
    BATCH = 8

    # Dummy Input
    dummy_input = torch.randn(BATCH, SEQ_LEN, DIM, device=device)

    # Initialize Models
    standard_model = StandardAttention(DIM, HEADS).to(device)
    optimized_model = CodexOptimizedAttention(DIM, HEADS).to(device)

    # Warmup Phase
    print("🔥 Warming up GPU...")
    _ = standard_model(dummy_input)
    _ = optimized_model(dummy_input)

    # Benchmark Phase
    print(f"📊 Benchmarking on Sequence Length: {SEQ_LEN}...")
    t_std = benchmark(standard_model, dummy_input)
    t_opt = benchmark(optimized_model, dummy_input)

    print(f"\nRESULTS:")
    print(f"🔴 Standard Implementation: {t_std:.2f} ms")
    print(f"🟢 Codex Optimized:        {t_opt:.2f} ms")
    print(f"🚀 Speedup Factor:         {t_std / t_opt:.2f}x Faster")
