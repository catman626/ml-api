import torch

a = torch.ones(size=(3,4,5))

for i in range(3):
    for j in range(4):
        for k in range(5):
            a[i][j][k] = i*100+j*10+k


# get 13, 22
idx1 = torch.tensor([1, 2])
idx2 = torch.tensor([3, 2])

b = a[idx1, idx2]
print(b)





