import torch

def flatten_then_reshape(x: torch.Tensor, new_shape) -> torch.Tensor:
    # TODO: flatten x to 1-D, then rearrange into new_shape
    x_flatten = x.flatten()
    x_reshape = x_flatten.reshape(new_shape)
    return x_reshape

def transpose_last_two(x: torch.Tensor) -> torch.Tensor:
    # TODO: swap the last two dimensions of x
    output = x.transpose(-1,-2)
    return output
