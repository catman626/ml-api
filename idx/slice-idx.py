import torch

a = torch.ones(size=(3,4,5))

for i in range(3):
    for j in range(4):
        for k in range(5):
            a[i][j][k] = i*100+j*10+k

literal_idx = a[:, :2]
tuple_slice = (slice(None), slice(0, 2))
tuple_sliced = a[tuple_slice]

print(f" >>>>>> a.shape: {a.shape}")
print(f" >>> a[:, :2]: {literal_idx.shape}")
print(f" >>> a[slice_tuple]: {tuple_sliced.shape}")

