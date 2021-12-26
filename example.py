import torch
from mi_lower_bounds import mi_lower_bound


# test scores
scores = torch.tensor([[0.42340052, 0.06299424],
                            [0.06570804, 0.14229548]])


if __name__ == "__main__":
	estimator = "nwj" # four possible options tuba, nwj, nce, interpolate
	true_value = (-0.1094836)
	estimate = mi_lower_bound(scores, estimator, mean_reduction=True)
	assert abs(estimate - true_value) <= 1e-6
	print("Passed!")