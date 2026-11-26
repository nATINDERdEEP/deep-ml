import numpy as np
def softmax(X, axis):
    e_x = np.exp(X- np.max(X, axis= axis, keepdims = True))
    total_sum = np.sum(e_x, axis= axis, keepdims = True)
    return e_x/total_sum

def multi_head_attention(X, weights, num_heads):
    B, T, D = X.shape
    d_h = D // num_heads

    W_q = weights["W_q"]
    W_k = weights["W_k"]
    W_v = weights["W_v"]
    W_o = weights["W_o"]

    query = X@W_q
    keys =  X@W_k
    values = X@W_v

    query =  query.reshape(B,T,num_heads,d_h)
    keys = keys.reshape(B,T,num_heads,d_h)
    values = values.reshape(B,T,num_heads,d_h)

    query =  query.transpose(0,2,1,3)
    keys =  keys.transpose(0,2,1,3)
    values = values.transpose(0,2,1,3)

    # new_Dim -> B,num_heads,T,d_h

    dot_prod = (query @ keys.transpose(0, 1, 3, 2)) / np.sqrt(d_h) 
    # new_Dim -> B,num_heads, T,T
    sftmax = softmax(dot_prod, axis = -1)

    output = sftmax@values
    output = output.transpose(0,2,1,3)
    output = output.reshape(B,T,D)
    attn_output = output@W_o 
    return attn_output

def add_norm_MHA(X, weights, num_heads, eps=1e-5):
    X = X + multi_head_attention(X, weights, num_heads)
    # X_dim -> (B, T, D)
    mean = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)
    norm = (X - mean) / np.sqrt(var + eps)
    output = weights['gamma1'] * norm + weights['beta1']
    return output

def feed_forward_Network(X, weights):
    layer_1 = X@weights['W1'] + weights['b1']
    activation = np.maximum(0,layer_1)
    layer_2 = activation@weights['W2'] + weights['b2']
    return layer_2

def add_norm_FFN(X, weights, eps=1e-5):
    X = X + feed_forward_Network(X, weights)
    # X_dim -> (B, T, D)
    mean = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)
    norm = (X - mean) / np.sqrt(var + eps)
    output = weights['gamma2'] * norm + weights['beta2']
    return output

def transformer_encoder_layer(X: np.ndarray, weights: dict, num_heads: int, eps: float = 1e-5) -> np.ndarray:
    """
    Forward pass of a single Transformer Encoder Layer.

    Args:
        X: Input tensor of shape (batch_size, seq_len, d_model)
        weights: Dictionary containing all weight matrices and normalization parameters
        num_heads: Number of attention heads
        eps: Epsilon for layer normalization

    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """
    out1 = add_norm_MHA(X, weights, num_heads, eps)
    out2 = add_norm_FFN(out1, weights, eps)

    return out2

