import torch
import flashinfer
kv_len = 4096
num_qo_heads = 32
num_kv_heads = 32
head_dim = 128



q = torch.randn(num_qo_heads, head_dim).half().to("cuda:0")
k = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
v = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")


# 
o = flashinfer.single_decode_with_kv_cache(q, k, v)
print(f" >>> o.shape: {o.shape}")
print(" >>> reference: torch.Size([32, 128])")
