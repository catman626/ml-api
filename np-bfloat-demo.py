import ml_dtypes
import numpy as np
import torch

a = np.array([3, 4, 5], dtype=ml_dtypes.bfloat16)
print(f" >>> shape of a: {a.shape}")

np.save("tmp1", a)


def get_test_data():
    return torch.rand(3, 4, 5, dtype=torch.bfloat16)

def bf16_torch_to_np(torch_data):
    return torch_data.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)

def bf16_np_to_torch(np_data):
    return torch.tensor(np_data.view(np.uint16)).view(torch.bfloat16)
    
def test_reversibility():
    b = get_test_data()
    np_b = bf16_torch_to_np(b)
    r_b = bf16_np_to_torch(np_b)
    diff = (r_b - b).max().item()
    print(f" >>> diff: {diff}")


def test_store_load():
    a = get_test_data()
    a_np = bf16_torch_to_np(a)

    np.save("tmp", a_np)
    b = np.load("tmp.npy")
    print(f" >>> shape of b: {b.shape}")
    


if __name__ == "__main__":
    test_reversibility()
    test_store_load()
