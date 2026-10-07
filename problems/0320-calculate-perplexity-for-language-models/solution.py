import numpy as np

def calculate_perplexity(probabilities: list[float]) -> float:
    """
    Calculate the perplexity of a language model given token probabilities.
    
    Args:
        probabilities: List of probabilities P(token_i | context) for each token
                      in the sequence, where each probability is in (0, 1]
    
    Returns:
        Perplexity value as a float
    """
    epsilon = np.finfo(np.float32).eps
    return np.exp(-1*np.mean([np.log(probabilities + epsilon) for probabilities in probabilities]))