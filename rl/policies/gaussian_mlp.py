import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from rl.distributions import DiagonalGaussian
from .running_stat import ObsNorm


class GaussianMLP(nn.Module):
    """
    Gaussian Multilayer Perceptron Policy.

    Policy that samples actions from a gaussian distribution with a
    learned (but state invariant) standard deviation.
    """

    def __init__(self, obs_dim, action_dim,  hidden_dims=(64, 64),
                 init_std=1.0, nonlin=F.tanh):

        super(GaussianMLP, self).__init__()

        def schulman_init(m):
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                m.weight.data.normal_(0, 1)
                m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
                if m.bias is not None:
                    m.bias.data.fill_(0)

        self.obs_filter = ObsNorm((1, obs_dim), clip=5)

        self.distribution = DiagonalGaussian()

        # create actor network
        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.Linear(obs_dim, hidden_dims[0])]
        for l in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[l]
            out_dim = hidden_dims[l + 1]
            self.actor_layers += [nn.Linear(in_dim, out_dim)]

        self.means = nn.Linear(hidden_dims[-1], action_dim)

        self.log_stds = nn.Parameter(
            torch.ones(1, action_dim) * np.log(init_std)
        )

        # create critic network
        self.critic_layers = nn.ModuleList()
        self.critic_layers += [nn.Linear(obs_dim, hidden_dims[0])]
        for l in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[l]
            out_dim = hidden_dims[l + 1]
            self.critic_layers += [nn.Linear(in_dim, out_dim)]

        self.vf = nn.Linear(hidden_dims[-1], 1)


        self.apply(schulman_init)
        self.means.weight.data.mul_(0.01)

        self.nonlin = nonlin

        self.train()

    def forward(self, inputs):
        inputs.data = self.obs_filter(inputs.data)

        x = inputs
        for l in self.critic_layers:
            x = self.nonlin(l(x))
        value = self.vf(x)

        x = inputs
        for l in self.actor_layers:
            x = self.nonlin(l(x))
        means = self.means(x)

        log_stds = self.log_stds.expand_as(means)

        return value, means, log_stds

    def act(self, observation, stochastic=True):
        value, action_means, action_log_stds = self(observation)
        action_stds = action_log_stds.exp()

        params = dict(
            mu=action_means,
            sigma=action_stds,
            log_sigma=action_log_stds
        )

        if not stochastic:
            action = action_means
        else:
            action = self.distribution.sample(params)

        return value, action

    def evaluate_actions(self, observations, actions):
        value, action_means, action_log_stds = self(observations)
        action_stds = action_log_stds.exp()

        params = dict(
            mu=action_means,
            sigma=action_stds,
            log_sigma=action_log_stds
        )

        action_log_probs = self.distribution.log_likelihood(actions, params)
        return value, action_log_probs, params
