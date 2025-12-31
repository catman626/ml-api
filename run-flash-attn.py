import torch
from flash_attn import flash_attn_func

# 初始化 Q/K/V（FP16，CUDA）
b, N, M, H, D = 2, 128, 16, 8, 64
q = torch.randn(b, N, H, D, dtype=torch.float16, device="cuda")
k = torch.randn(b, M, H, D, dtype=torch.float16, device="cuda")
v = torch.randn(b, M, H, D, dtype=torch.float16, device="cuda")

# 调用 FlashAttention v2（因果掩码）
output = flash_attn_func(
    q, k, v,
    dropout_p=0.1,
    causal=True,  # 启用因果掩码（仅允许关注前面的token）
)

print("原生 FlashAttention 输出形状:", output.shape)  # [2, 128, 8, 64]
