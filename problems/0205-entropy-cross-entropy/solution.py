import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	"""
	Compute entropy of P and cross-entropy between P and Q.
	
	Args:
		P: True probability distribution
		Q: Predicted probability distribution
	
	Returns:
		Tuple of (entropy H(P), cross-entropy H(P,Q))
	"""
	entropy = 0
	cross_entropy = 0
	epsilon = np.finfo(np.float64).eps
	for index in range(len(P)):
		entropy += P[index]*np.log(P[index] + epsilon)
		cross_entropy += P[index]*np.log(Q[index] + epsilon)
	
	return (-1*entropy,-1*cross_entropy)