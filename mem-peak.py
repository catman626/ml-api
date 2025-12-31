import torch

a = torch.rand(8, 14, 16000, 64, device="cuda", dtype=torch.float)
b = torch.rand(8, 14, 1600, 64, device="cuda", dtype=torch.float)

c = torch.matmul(a, b.transpose(2, 3))

print(f" >>> shape of c: {c.shape}")

peak = torch.cuda.max_memory_allocated("cuda") // 2**30

print(f" >>> peak memory usage: {peak} GB")