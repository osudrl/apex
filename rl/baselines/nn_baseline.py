import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from rl.distributions import DiagonalGaussian


class Critic(nn.Module):

    def __init__(self, obs_dim, hidden_dims=(64, 64), nonlin=F.tanh):

        super(Critic, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers += [nn.Linear(obs_dim, hidden_dims[0])]

        for l in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[l]
            out_dim = hidden_dims[l + 1]

            self.hidden_layers += [nn.Linear(in_dim, out_dim)]

        self.vf = nn.Linear(hidden_dims[-1], 1) # value function estimator

        self.nonlin = nonlin

    def forward(self, x):
        output = x
        for l in self.hidden_layers:
            output = self.nonlin(l(output))
        critic = self.vf(output)

        return critic
