"""Implements conservative policy iteration."""
from copy import deepcopy

import torch
import torch.optim as optim

from rl.utils import center
from rl.baselines import ZeroBaseline
from ..base import PolicyGradientAlgorithm

import numpy as np


class CPI(PolicyGradientAlgorithm):
    def __init__(self, env, policy, tau=0.97, discount=0.99, lr=0.01,
                 baseline=None):
        self.env = env
        self.policy = policy
        self.discount = discount
        self.tau = tau

        self.optimizer = optim.Adam(policy.parameters(), lr=lr)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--n_itr", type=int, default=1000,
                            help="number of iterations of the learning algorithm")
        parser.add_argument("--max_trj_len", type=int, default=200,
                            help="maximum trajectory length")
        parser.add_argument("--n_trj", type=int, default=100,
                            help="number of sample trajectories per iteration")
        parser.add_argument("--lr", type=int, default=0.01,
                            help="Adam learning rate")
        parser.add_argument("--desired_kl", type=int, default=0.01,
                            help="Desired change in mean kl per iteration")
        parser.add_argument("--tau", type=int, default=0.97,
                            help="Generalized advantage estimate discount")

    def train(self, n_itr, n_trj, max_trj_len, adaptive=True, desired_kl=2e-3,
              explore_bonus=1e-4, logger=None):
        env = self.env
        policy = self.policy
        for _ in range(n_itr):
            paths = [self.rollout(env, policy, max_trj_len) for _ in range(n_trj)]

            observations = torch.cat([p["observations"] for p in paths])
            actions = torch.cat([p["actions"] for p in paths])
            advantages = torch.cat([p["advantages"] for p in paths])
            returns = torch.cat([p["returns"] for p in paths])

            advantages = center(advantages)

            old_policy = deepcopy(policy)

            pd = policy.get_pdparams(observations)
            old_pd = old_policy.get_pdparams(observations)

            entropy = policy.distribution.entropy(pd)

            ratio = policy.distribution.likelihood_ratio(
                actions,
                old_pd,
                pd
            )

            ratio = policy.distribution.log_likelihood(actions, pd)

            policy_loss = -(ratio * advantages).mean()
            entropy_penalty = -(explore_bonus * entropy).mean()

            total_loss = policy_loss + entropy_penalty

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            for _ in range(10):
                critic = policy.get_critic(observations)
                self.optimizer.zero_grad()
                critic_loss = (critic - returns).pow(2).mean()
                critic_loss.backward()
                self.optimizer.step()

            """
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
                    lr_multiplier = 1.0

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= lr_multiplier
            """

            mean_reward = np.mean(([p["rewards"].sum().data[0] for p in paths]))
            mean_entropy = entropy.mean().data[0]

            print(mean_reward)

            if logger is not None:
                logger.record("Reward", mean_reward)
                #logger.record("Mean KL", mean_kl)
                #logger.record("Entropy", mean_entropy)
                logger.dump()
