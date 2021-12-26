import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# This function calculates the mean of the off-diagonal terms in log space
def reduce_logmeanexp_off_diag(x, dim=None):
    batch_size = x.shape[0]
    if dim == None:
        dim = tuple(i for i in range(x.dim()))
    # calculate the sum of the log of the sum of the exponentiated terms
    logsumexp = torch.logsumexp(x - torch.diag( np.inf * torch.ones(batch_size) ), dim=dim)
    if dim == 0:
        num_elem = batch_size - 1
    else:
        num_elem = batch_size * (batch_size - 1)
    return logsumexp - torch.log(torch.tensor(num_elem))


   # Interpolated lower bound I_{\alpha}
def log_interpolate(log_a, log_b, alpha_logit):
    '''
    Numerically stable interpolation in log space: log( alpha * a + (1-alpha) * b  )
    '''
    log_alpha =  - F.softplus(torch.tensor(-alpha_logit))
    log_1_minus_alpha =  - F.softplus(torch.tensor(alpha_logit))
    dim = tuple(i for i in range(log_a.dim()))
    y = torch.logsumexp( torch.stack((log_alpha + log_a, log_1_minus_alpha + log_b)) , dim=0)
    return y

def compute_log_loomean(scores):
    '''
    comute the mean of the second term of the interpolated lower bound (leave-one-out) in a numerically stable way
    '''
    max_score = torch.max(scores, dim=-1, keepdim=True)[0]
    # logsumexp minus the max
    lse_minus_max = torch.logsumexp(scores - max_score, dim=1, keepdim=True)
    d = lse_minus_max + (max_score - scores)
    d_ok = torch.not_equal(d, 0.)
    safe_d = torch.where(d_ok, d, torch.ones_like(d))
    loo_lse = scores + softplus_inverse(safe_d)
    # normailize by the batch size
    loo_lme = loo_lse - torch.log(torch.tensor(scores.shape[1]) - 1.)
    return loo_lme

def softplus_inverse(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    '''
    A function that implement tge softplus invserse, this is based on tensorflow implementatiion on 
    https://github.com/tensorflow/probability/blob/v0.15.0/tensorflow_probability/python/math/generic.py#L494-L545
    '''
    # check the limit for floating point arethmatics in numpy
    threshold = np.log(np.finfo(np.float32).eps) + 2.
    is_too_small =  x < np.exp(threshold)
    is_too_large = x > -threshold
    too_small_value = torch.log(x)
    too_large_value = x
    x = torch.where(is_too_small | is_too_large, torch.ones_like(x), x)
    y = x + torch.log(-(torch.exp(-x) - 1))
    return torch.where(is_too_small, too_small_value, torch.where(is_too_large, too_large_value, y))