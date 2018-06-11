import math
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def truncated_normal_init(vocab_size, embed_size, rng):
    values = np.zeros((vocab_size, embed_size))
    scale=1.0/math.sqrt(embed_size)
    for i in range(vocab_size):
        for j in range(embed_size):
            while (True):
                value = rng.normal(scale=scale)
                # if (abs(value) <= 2 * scale):
                break
            values[i, j] = value

    return values


def to_value(x):
    return x.data.cpu().numpy()


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def get_torch_var(x, typ = 'Long'):
    if (typ == 'Long'):
        return Variable(torch.from_numpy(np.asarray(x, dtype=np.int64)), requires_grad=False).cuda()
    elif (typ == 'Float'):
        return Variable(torch.from_numpy(np.asarray(x, dtype=np.float32)), requires_grad=False).cuda()
    # else:
    #     return Variable(torch.cuda.LongTensor(x), requires_grad=False)


def sim_score(x, y, measure='Gaussian'):
    if (measure == 'Gaussian'):
        size = len(x) / 2
        x_mu, x_sigma  = x[:size], x[size:]
        y_mu, y_sigma  = y[:size], y[size:]
        score = 0
        for mu1, sigma1, mu2, sigma2 in zip(x_mu, x_sigma, y_mu, y_sigma):
            score += sigma2/sigma1 + (mu1-mu2)*(mu1-mu2)/sigma1 - math.log(sigma2/sigma1)
    else:
        score = np.dot(x, y)

    return score

def gaussian_sim(mu_x, log_sigma_x, mu_y, log_sigma_y):
    batch, sampled, size = mu_y.size(0), mu_y.size(1), mu_y.size(2)
    sigma_x = log_sigma_x.exp()
    sigma_y = log_sigma_y.exp()

    mu_diff = mu_x.view(-1, 1, size).expand_as(mu_y) - mu_y
    part1 = torch.bmm(
        sigma_y,
        1 / sigma_x.view(-1, size, 1)).view(-1, sampled)
    part2 = torch.sum(mu_diff 
        / sigma_x.view(-1, 1, size).expand_as(mu_y)
        * mu_diff, 2).view(-1, sampled)
    part3 = -size
    part4 = ((torch.sum(log_sigma_x.view(-1, 1, size).expand_as(mu_y), 2)
                -torch.sum(log_sigma_y, 2))
            .view(-1, sampled))
    return -(part1 + part2 + part3 + part4) / 2
    
    
