import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    z_max = max(scores)
    result = []
    for Z_i in scores:
        result.append(np.exp(Z_i - z_max))
    
    sum_result = sum(result)
    return result/sum_result
    