import torch

def make_tensor():
    """Return a 2x3 float32 tensor [[1, 2, 3], [4, 5, 6]]."""
    list_ = [[1,2,3],[4,5,6]]
    return torch.tensor(list_, dtype= torch.float32)

