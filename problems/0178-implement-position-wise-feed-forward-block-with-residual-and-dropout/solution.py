import numpy as np

def ffn(x: list[float], W1: list[list[float]], b1: list[float], W2: list[list[float]], b2: list[float], dropout_p: float=0.1, seed: int=42) -> list[float]:
    """
    Implement a position-wise feed-forward block with residual and dropout.

    Args:
        x: input vector
        W1, b1: first linear layer parameters
        W2, b2: second linear layer parameters
        dropout_p: dropout probability
        seed: random seed for reproducibility

    Returns:
        Output vector after FFN block (rounded to 4 decimals)
    """
    X, W1, b1, W2, b2 = map(np.array, (x, W1, b1, W2, b2))

    # First linear layer
    layer_1 = W1 @ X + b1

    # ReLU
    layer_2 = np.maximum(0, layer_1)

    # Second linear layer
    layer_3 = W2 @ layer_2 + b2

    # Inverted dropout
    np.random.seed(seed)
    mask = np.random.rand(*layer_3.shape) > dropout_p
    y = (layer_3 * mask) / (1 - dropout_p)

    # Residual connection
    output = y + X

    return np.round(output, 4).tolist()
