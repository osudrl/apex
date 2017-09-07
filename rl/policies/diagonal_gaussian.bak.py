from rl.distributions.base import Distribution
from torch.autograd import Variable
import torch
import numpy as np


class DiagonalGaussian(Distribution):
    def __init__(self, mu=None, sigma=None, log_sigma=None):
        self.mu = mu
        self.sigma = sigma
        self.log_sigma = log_sigma

    def log_likelihood(self, x):
        var = self.sigma.pow(2)

        log_density = -(x - self.mu).pow(2) / (
            2 * var) - 0.5 * np.log(2 * np.pi) - self.log_sigma

        return log_density.sum(1)

    def kl_divergence(self, new):
        kl = new.log_sigma - self.log_sigma + \
             (self.sigma.pow(2) + (self.mu - new.mu).pow(2)) / \
             (2.0 * new.sigma.pow(2)) - 0.5
        return kl.sum(1)

    def entropy(self):
        return (self.log_sigma + np.log(np.sqrt(2 * np.pi * np.e))).sum(1)

    def sample(self):
        action = torch.normal(self.mu, self.sigma)
        return action.detach()

    def copy(self):
        return DiagonalGaussian(
            self.mu.detach(),
            self.sigma.detach(),
            self.log_sigma.detach()
        )
