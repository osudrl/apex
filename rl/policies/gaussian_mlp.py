import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from distributions.diagonal_gaussian import DiagonalGaussian


class GaussianMLP(nn.Module):
    """
    Gaussian Multilayer Perceptron Policy.

    Policy that samples actions from a gaussian distribution with a
    learned (but state invariant) standard deviation.
    """

    def __init__(self, obs_dim, action_dim,  hidden_dims=(32, 32),
                 init_std=1.0, nonlin=F.tanh, optimizer=optim.Adam):

        super(GaussianMLP, self).__init__()

        self.distribution = DiagonalGaussian()

        self.hidden_layers = nn.ModuleList()

        self.hidden_layers += [nn.Linear(obs_dim, hidden_dims[0])]

        for l in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[l]
            out_dim = hidden_dims[l + 1]
            self.hidden_layers += [nn.Linear(in_dim, out_dim)]

        self.out = nn.Linear(hidden_dims[-1], action_dim)

        self.log_stds = nn.Parameter(
            torch.ones(1, action_dim) * np.log(init_std)
        )

        self.nonlin = nonlin

    def forward(self, x):
        for l in self.hidden_layers:
            x = self.nonlin(l(x))

        means = self.out(x)

        log_stds = self.log_stds.expand_as(means)

        stds = torch.exp(log_stds)

        return means, log_stds, stds

    def _update(self, observations):
        means, log_stds, stds = self(observations)

        self.distribution.mu = means
        self.distribution.log_sigma = log_stds
        self.distribution.sigma = stds

    def get_action(self, obs, stochastic=True):
        self._update(obs)

        if not stochastic:
            return self.distribution.mu

        return self.distribution.sample()

    def get_distribution(self, observations):
        self._update(observations)

        return self.distribution
