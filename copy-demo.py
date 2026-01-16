import torch

shape = (3, 4, 5)
buf = torch.empty(shape, dtype=torch.float16, pin_memory=True)

idx = torch.full(shape, 100000)


buf.copy_(idx)

print(f" >>> buf: {buf}")
print(f" >>> idx: {idx}")


