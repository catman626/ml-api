import torch


dtype, elem_sz= torch.float32, 4
b, s, H = 32, 2048, 1024

A = torch.rand(b, s, H, device="cuda", dtype=dtype)
B = torch.rand(H, H, device="cuda", dtype=dtype)

c = torch.matmul(A, B)

print(f" >>> shape of c: {c.shape}")

estimation = (2* b*s*H + H*H ) * elem_sz / 2**30
peak = torch.cuda.max_memory_allocated("cuda") / 2**30

print(f" >>> peak memory usage: {peak} GB")
print(f" >>> estimation: {estimation} GB")
