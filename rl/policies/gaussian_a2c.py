import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from .running_stat import ObsNorm

import numpy as np

from rl.distributions import DiagonalGaussian


class GaussianA2C(nn.Module):
    """
    Gaussian Multilayer Perceptron Policy.

    Policy that samples actions from a gaussian distribution with a
    learned (but state invariant) standard deviation.

    Contains neural network critic with same structure as actor.
    """

    def __init__(self, obs_dim, action_dim,  hidden_dims=(32, 32),
                 init_std=1.0, nonlin=F.tanh):

        super(GaussianA2C, self).__init__()

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                nn.init.orthogonal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0)

        self.obs_filter = ObsNorm((1, obs_dim), clip=5)
        self.critic = None

        self.distribution = DiagonalGaussian()

        self.hidden_layers_actor = nn.ModuleList()
        self.hidden_layers_actor += [nn.Linear(obs_dim, hidden_dims[0])]

        self.hidden_layers_critic = nn.ModuleList()
        self.hidden_layers_critic += [nn.Linear(obs_dim, hidden_dims[0])]

        for l in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[l]
            out_dim = hidden_dims[l + 1]

            self.hidden_layers_actor += [nn.Linear(in_dim, out_dim)]
            self.hidden_layers_critic += [nn.Linear(in_dim, out_dim)]

        self.means = nn.Linear(hidden_dims[-1], action_dim)

        self.log_stds = nn.Parameter(
            torch.ones(1, action_dim) * np.log(init_std)
        )

        self.vf = nn.Linear(hidden_dims[-1], 1) # value function estimator

        self.apply(weights_init)

        self.nonlin = nonlin

    def forward(self, x):
        x.data = self.obs_filter(x.data)
        output = x
        for l in self.hidden_layers_actor:
            output = self.nonlin(l(output))
        means = self.means(output)

        output = x
        for l in self.hidden_layers_critic:
            output = self.nonlin(l(output))
        critic = self.vf(output)

        log_stds = self.log_stds.expand_as(means)

        stds = torch.exp(log_stds)

        return means, log_stds, stds, critic

    def get_action(self, obs, stochastic=True):
        means, log_stds, stds, _ = self(obs)

        params = dict(
            mu=means,
            sigma=stds,
            log_sigma=log_stds
        )

        if not stochastic:
            return params["mu"]

        return self.distribution.sample(params)

    def get_pdparams(self, observations):
        means, log_stds, stds, _ = self(observations)

        return dict(
            mu=means,
            sigma=stds,
            log_sigma=log_stds
        )

    def get_critic(self, states):
        _, _, _, critic = self(states)
        return critic
