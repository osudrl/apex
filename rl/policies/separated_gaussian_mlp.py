import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.distributions import DiagonalGaussian
from .base import FFPolicy

import time

# NOTE: the fact that this has the same name as a parameter caused a NASTY bug
# apparently "if <function_name>" evaluates to True in python...
def normc_fn(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

# distributional TD3 critic that uses mixture of gaussians (D4PG)
# action is not included until the 2nd layer of the critic (from paper)
class LN_Distrib_TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size1, hidden_size2):
        super(LN_Distrib_TD3Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim)


class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()
        self.env = None # Gym environment name string

    def forward(self, x):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        value, x = self(inputs)
        action = self.dist.sample(x, deterministic=deterministic)
        return value, action.detach()

    def evaluate(self, inputs):
        value, x = self(inputs)
        return value, self.dist.evaluate(x)


class GaussianMLP(FFPolicy):
    def __init__(self, 
                 num_inputs, 
                 action_dim, 
                 init_std=1, 
                 learn_std=True, 
                 nonlinearity="tanh", 
                 bounded=False,
                 normc_init=True,
                 obs_std=None,
                 obs_mean=None):
        super(GaussianMLP, self).__init__()

        actor_dims = (256, 256)
        critic_dims = (256, 256)

        # create actor network
        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.Linear(num_inputs, actor_dims[0])]
        for l in range(len(actor_dims) - 1):
            in_dim = actor_dims[l]
            out_dim = actor_dims[l + 1]
            self.actor_layers += [nn.Linear(in_dim, out_dim)]
        
        self.mean = nn.Linear(actor_dims[-1], action_dim)

        # create critic network
        self.critic_layers = nn.ModuleList()
        self.critic_layers += [nn.Linear(num_inputs, critic_dims[0])]
        for l in range(len(critic_dims) - 1):
            in_dim = critic_dims[l]
            out_dim = critic_dims[l + 1]
            self.critic_layers += [nn.Linear(in_dim, out_dim)]

        self.vf = nn.Linear(critic_dims[-1], 1)

        self.dist = DiagonalGaussian(action_dim, init_std, learn_std)

        if nonlinearity == "relu":
            self.nonlinearity = F.relu
        elif nonlinearity == "tanh":
            self.nonlinearity = torch.tanh
        else:
            raise NotImplementedError

        self.obs_std = obs_std
        self.obs_mean = obs_mean
        
        # weight initialization scheme used in PPO paper experiments
        self.normc_init = normc_init

        self.bounded = bounded

        self.init_parameters()
        self.train()

    def init_parameters(self):
        if self.normc_init:
            print("Doing norm column initialization.")
            self.apply(normc_fn)

            if self.dist.__class__.__name__ == "DiagGaussian":
                self.mean.weight.data.mul_(0.01)

    def forward(self, inputs):
        if self.training == False:
            inputs = (inputs - self.obs_mean) / self.obs_std

        x = inputs
        for l in self.critic_layers:
            x = self.nonlinearity(l(x))
        value = self.vf(x)

        x = inputs
        for l in self.actor_layers:
            x = self.nonlinearity(l(x))
        x = self.mean(x)

        if self.bounded:
            mean = torch.tanh(x) 
        else:
            mean = x

        return value, mean