import numpy as np

def compute_cross_entropy_loss(
    predicted_probs: np.ndarray,
    true_labels: np.ndarray,
    epsilon=1e-15
) -> float:
    
    true_class = []

    for k in range(len(predicted_probs)):
        for index in range(len(true_labels[k])):
            if true_labels[k][index] == 1:
                true_class.append(
                    np.log(predicted_probs[k][index] + epsilon)
                )

    return -np.mean(true_class)