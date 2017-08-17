"""Implementation of vanilla policy gradient."""
import torch
import torch.optim as optim
from torch.autograd import Variable

from rl.utils import center
from rl.baselines import ZeroBaseline
from ..base import PolicyGradientAlgorithm

import numpy as np


class VPG(PolicyGradientAlgorithm):
    """
    Implements vanilla policy gradient aka REINFORCE.

    See: http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf

    Includes adaptive learning rate implementation mentioned in
    Schulman et al 2017. See: https://arxiv.org/pdf/1707.06347.pdf

    Includes entropy bonus mentioned in Mnih et al 2016.

    Includes GAE described in Schulman et al 2016.
    """

    def __init__(self, env, policy, discount=0.99, lr=0.01, baseline=None):
        self.env = env
        self.policy = policy
        self.discount = discount

        if baseline is None:
            self.baseline = ZeroBaseline()
        else:
            self.baseline = baseline

        self.optimizer = optim.Adam(policy.parameters(), lr=lr)

    def train(self, n_itr, n_trj, max_trj_len, adaptive=True, desired_kl=2e-3,
              explore_bonus=1e-4):
        env = self.env
        policy = self.policy
        for _ in range(n_itr):
            paths = [self.rollout(env, policy, max_trj_len) for _ in range(n_trj)]

            observations = torch.cat([p["observations"] for p in paths])
            actions = torch.cat([p["actions"] for p in paths])
            advantages = torch.cat([p["advantages"] for p in paths])

            # TODO: verify centering advantages over whole batch instead of per
            # path actually makes sense
            advantages = center(advantages)

            distribution = policy.get_distribution(observations)

            logprobs = distribution.log_likelihood(actions)
            entropy = distribution.entropy()

            policy_loss = logprobs * advantages
            total_loss = -(policy_loss + explore_bonus * entropy).mean()

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            self.baseline.fit(paths)

            old_distribution = distribution.copy()
            new_distribution = policy.get_distribution(observations)

            kl_oldnew = old_distribution.kl_divergence(new_distribution)

            mean_kl = kl_oldnew.mean().data[0]

            # see: https://arxiv.org/pdf/1707.06347.pdf, footnote 3
            if adaptive:
                if mean_kl > desired_kl * 2:
                    lr_multiplier = 1 / 1.5
                elif mean_kl < desired_kl / 2:
                    lr_multiplier = 1 * 1.5
                else:
                    lr_multiplier = 1

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= lr_multiplier

            print(
                'Average Return: %f, iteration: %d' %
                (np.mean(([p["rewards"].sum().data.numpy() for p in paths])), _)
            )
