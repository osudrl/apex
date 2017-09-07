# insert this to the top of your scripts (usually main.py)
import sys, warnings, traceback, torch
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
    traceback.print_stack(sys._getframe(2))
warnings.showwarning = warn_with_traceback; warnings.simplefilter('always', UserWarning);
torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn.functional as F


import numpy as np
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np

from rl.distributions import DiagonalGaussian
from rl.policies import GaussianMLP
from rl.utils import center, RLDataset


class Critic(nn.Module):

    def __init__(self, obs_dim, hidden_dims=(32,), nonlin=F.tanh):

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


if __name__ == "__main__":
    env = gym.make("Hopper-v1")

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    critic = Critic(obs_dim)
    policy = GaussianMLP(obs_dim, action_dim)

    optimizer = Adam(
        list(critic.parameters()) + list(policy.parameters()),
        lr=3e-4
    )

    for t in range(10):
        print("======== %d ========" % t)
        observations = []
        rewards = []
        values = []
        returns = []
        advantages = []
        actions = []

        obs = env.reset().ravel()[None, :]
        obs_var = Variable(torch.Tensor(obs))

        values.append(critic(obs_var))
        for _ in range(1000):
            action = policy.get_action(obs_var)

            next_obs, reward, done, _ = env.step(action.data.numpy().ravel())

            observations.append(obs_var)
            actions.append(action)
            rewards.append(Variable(torch.Tensor([[reward]])))

            obs = next_obs.ravel()[None, :]
            obs_var = Variable(torch.Tensor(obs))

            values.append(critic(obs_var))

        R = Variable(torch.zeros(1, 1))
        advantage = Variable(torch.zeros(1, 1))
        for t in reversed(range(len(rewards))):
            R = .99 * R + rewards[t]

            delta = rewards[t] + .99 * values[t + 1] - values[t]
            advantage = advantage * .99 * .95 + delta

            returns.append(R)
            advantages.append(advantage)

        returns = torch.cat(returns[::-1])
        advantages = torch.cat(advantages[::-1])
        observations = torch.cat(observations)
        actions = torch.cat(actions)

        dataset = RLDataset(observations, actions, advantages, returns)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        for e in range(10):
            losses = []
            for batch in dataloader:
                obs, ac, adv, ret = map(Variable, batch)

                optimizer.zero_grad()

                c_loss = (critic(obs) - ret).pow(2).mean()
                losses.append(c_loss.data[0])

                pd_params = policy.get_pdparams(obs)
                pi_loss = policy.distribution.log_likelihood(ac, pd_params) * adv


                loss = pi_loss.mean() #+ c_loss
                loss.backward()
                optimizer.step()
            print(sum(losses)/len(losses))
