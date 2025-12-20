import torch


a = torch.ones(3, 4, 5)
b = torch.ones(6,)

b = b.expand(3, 4, 6)

c = torch.concat([a, b], dim=-1)

print(f" >>> concat shape: {c.shape}")


