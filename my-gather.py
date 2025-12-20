import torch

# 示例数据
B, H, S, D, k = 2, 4, 10, 8, 3
kvcache = torch.randn(B, H, S, D)          # (2, 4, 10, 8)
# indices = torch.randint(0, S, (B, H, k))    # (2, 4, 3)
# selected = torch.gather(kvcache, dim=2, index=indices.unsqueeze(-1).expand(-1, -1, -1, D))

indices = torch.randint(0, S, (B, H, k, 1))    # (2, 4, 3)
selected = torch.gather(kvcache, dim=2, index=indices.expand(-1, -1, -1, D))

# 关键：在 dim=2（seq 维度）上 gather
# 注意：indices 需要扩展到和 kvcache 最后一维对齐

print(selected.shape)  # (2, 4, 3, 8)