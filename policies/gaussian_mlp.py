import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

import math

# MLP policy that outputs a mean and samples actions from a gaussian
# distribution with a learned (but state invariant) std dev
class GaussianMLP(nn.Module):
    def __init__(self, obs_dim, action_dim,  hidden_dims=(128, 128),
                 init_std=0.0, nonlin=F.tanh, optimizer=optim.Adam):

        super(GaussianMLP, self).__init__()

        self.hidden_layers = nn.ModuleList()

        self.hidden_layers += [nn.Linear(obs_dim, hidden_dims[0])]
        for dim in hidden_dims:
            self.hidden_layers += [nn.Linear(dim, dim)]

        self.out = nn.Linear(hidden_dims[-1], action_dim)

        self.log_stds = nn.Parameter(
            torch.ones(1, action_dim) * init_std
        )

        self.nonlin = nonlin

        self.optimizer = optimizer(self.parameters(),
                                   lr=0.003,
                                   weight_decay=0.0)

    def forward(self, x):
        for l in self.hidden_layers:
            x = self.nonlin(l(x))

        means = self.out(x)

        log_stds = self.log_stds.expand_as(means)

        stds = torch.exp(log_stds)

        return means, log_stds, stds

    def log_likelihood(self, x, means, log_stds, stds):
        var = stds.pow(2)

        log_density = -(x - means).pow(2) / (
            2 * var) - 0.5 * math.log(2 * math.pi) - log_stds

        return log_density.sum(1)

    def get_action(self, means, stds):
        action = torch.normal(means, stds)
        return action.detach()


    def loss(self, X, y):
        output = self.forward(X)
        loss = nn.MSELoss()(output, y)
        return loss

    def fit(self, X, y):
        self.optimizer.zero_grad()
        output = self.forward(X)
        loss = nn.MSELoss()(output, y)
        loss.backward()  # accumulate gradients
        self.optimizer.step()  # update parameters

        return loss.data
