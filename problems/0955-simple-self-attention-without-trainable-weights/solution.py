
import numpy as np

def simple_self_attention(X: list[list[float]]) -> list[list[float]]:
    query = np.array(X, dtype=float)
    key = np.array(X, dtype=float)
    values = np.array(X, dtype=float)

    dot_product = np.dot(query, key.T)

    for row in range(len(dot_product)):
        total_exp = 0
        row_max = np.max(dot_product[row])

        for col in range(len(dot_product[row])):
            total_exp += np.exp(dot_product[row][col] - row_max)

        for col in range(len(dot_product[row])):
            dot_product[row][col] = (
                np.exp(dot_product[row][col] - row_max) / total_exp
            )

    return np.dot(dot_product, values).tolist()
