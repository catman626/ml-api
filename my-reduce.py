import torch

a = torch.ones(3, 4, 5)

b = a.mean(dim=1)

print(b)
print(b.shape)
