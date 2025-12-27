import torch


a = torch.arange(10)
print(f" >>> shape of a: {a.shape}")

# index like this will expand the 
b = a[None, None, :, None]
print(f" >>> shape of b: {b.shape}")
