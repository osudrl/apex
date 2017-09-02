from rl.distributions.base import Distribution
from torch.autograd import Variable
import torch
import numpy as np


class DiagonalGaussian(Distribution):
    def log_likelihood(self, action, params):
        var = params["sigma"].pow(2)

        log_density = -(action - params["mu"]).pow(2) / (
            2 * var) - 0.5 * np.log(2 * np.pi) - params["log_sigma"]

        return log_density.sum(1, keepdim=True)

    def kl_divergence(self, old, new):
        kl = new["log_sigma"] - old["log_sigma"] + \
             (old["sigma"].pow(2) + (old["mu"] - new["mu"]).pow(2)) / \
             (2.0 * new["sigma"].pow(2)) - 0.5

        return kl.sum(1, keepdim=True)

    def entropy(self, params):
        return (params["log_sigma"] + np.log(np.sqrt(2 * np.pi * np.e))).sum(1, keepdim=True)

    def sample(self, params):
        action = torch.normal(params["mu"], params["sigma"])
        return action.detach()
