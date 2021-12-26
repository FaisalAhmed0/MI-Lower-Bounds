import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils import reduce_logmeanexp_off_diag
from utils import log_interpolate
from utils import compute_log_loomean
from utils import softplus_inverse



def mi_lower_bound(scores, estimator, log_baseline=None, alpha_logit=None, mean_reduction=False):
	'''
	This function calculate an estimate (lower bound) on the mutual information
	scores: scores output from a critic (B,B)
	estimator: lower bound estimator: "tuba", "nwj", "nce", "interpolate"
	log_baseline: log of the baseline term in "tuba" and "interpolate" lower bounds
	alpha_logit: alpha logit for "interpolate" lower bound
	'''
	estimators = {
		'tuba': tuba_lower_bound,
		'nwj' : nwj_lower_bound,
		'nce' : infoNCE_lower_bound,
		'interpolate': interpolated_lower_bound,
	}
	if estimator in estimators:
		if estimator == "tuba":
			return estimators["tuba"](scores, log_baseline, mean_reduction)
		elif estimator == "nwj":
			return estimators["nwj"](scores, mean_reduction)
		elif estimator == "nce":
			return estimators["nce"](scores, mean_reduction)
		elif estimator == "interpolate":
			return estimators["interpolate"](scores, log_baseline, alpha_logit, mean_reduction) 
		else:
			raise ValueError(f'{estimator} is not implemented')
	else:
		raise ValueError(f'{estimator} is Not found')


# I_{UTBA}
def tuba_lower_bound(scores, log_baseline=None, mean_reduction=False):
    if log_baseline is not None:
        scores -= log_baseline
    batch_size = float(scores.shape[0])
    # Expectation under the joint distribution
    if mean_reduction:
        joint_term = scores.diag().mean()
    else:
        joint_term = scores.diag()
    # Expecation under the marginals
    dim = None if mean_reduction else 0
    marginal_term = torch.exp(reduce_logmeanexp_off_diag(scores, dim=dim))
    return 1 + joint_term - marginal_term

# I_{NWJ}
def nwj_lower_bound(scores, mean_reduction=False):
    return tuba_lower_bound(scores - 1, mean_reduction=mean_reduction)

# I_{NCE} lower bound, biased but low variance
def infoNCE_lower_bound(scores, mean_reduction=False):
    batch_size = float(scores.shape[0])
    reduction = 'mean' if mean_reduction else 'none'
    nll = - nn.CrossEntropyLoss(reduction=reduction)(scores, target=torch.arange(int(batch_size)))
    mi = torch.log(torch.tensor(batch_size)) + nll
    return mi

# I_{\alpha}
def interpolated_lower_bound(scores, baseline, alpha_logit, mean_reduction=False):
    batch_size = scores.shape[0]
    # InfoNCE baseline
    infoNCE_baseline = compute_log_loomean(scores)
    # Interpolated baseline
    interpolated_baseline = log_interpolate(infoNCE_baseline, 
        torch.tile(baseline[:, None], (1, batch_size)), alpha_logit)
    # Marginal term
    critic_marg = scores - interpolated_baseline.diag()[:, None]
    dim = None if mean_reduction else 0
    marg_term =torch.exp(reduce_logmeanexp_off_diag(critic_marg, dim=dim))
    # Joint term
    critic_joint = scores.diag()[:, None] - interpolated_baseline
    if mean_reduction:
        joint_term = ( torch.sum(critic_joint) - torch.sum(critic_joint.diag())) / ( batch_size * (batch_size - 1.) )
    else:
        joint_term = ((critic_joint - torch.diag(critic_joint.diag()) ) /(batch_size - 1.)).sum(dim=0)
    return 1. + joint_term - marg_term


