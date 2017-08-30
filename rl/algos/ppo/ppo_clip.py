"""Proximal Policy Optimization with the clip objective."""
"""Implements conservative policy iteration."""
import torch
import torch.optim as optim

from rl.utils import center
from rl.baselines import ZeroBaseline
from ..base import PolicyGradientAlgorithm

import numpy as np


class PPO(PolicyGradientAlgorithm):

    def __init__(self, env, policy, discount=0.99, lr=0.01, baseline=None):
        self.env = env
        self.policy = policy
        self.discount = discount

        if baseline is None:
            self.baseline = ZeroBaseline()
        else:
            self.baseline = baseline

        self.optimizer = optim.Adam(policy.parameters(), lr=lr)


    def train(self, n_itr, n_trj, max_trj_len, epochs=10,
              explore_bonus=1e-4, logger=None):
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
            old_distribution = distribution.copy()

            entropy = distribution.entropy()

            # TODO: make ratio explicit, remove likelihood_ratio function?
            ratio = old_distribution.likelihood_ratio(
                actions,
                distribution
            )

            cpi_loss = ratio * advantages
            clip_loss = ratio.clamp(1.0 - epsilon, 1.0 + epsilon) * advantages

            ppo_loss = torch.minimum(cpi_loss, clip_loss)

            total_loss = -(ppo_loss + explore_bonus * entropy).mean()

            #for _ in range(epochs):
            #    

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            self.baseline.fit(paths)

            new_distribution = policy.get_distribution(observations)

            kl_oldnew = old_distribution.kl_divergence(new_distribution)

            mean_kl = kl_oldnew.mean().data[0]


            mean_reward = np.mean(([p["rewards"].sum().data[0] for p in paths]))
            mean_entropy = entropy.mean().data[0]

            if logger is not None:
                logger.record("Reward", mean_reward)
                logger.record("Mean KL", mean_kl)
                logger.record("Entropy", mean_entropy)
                logger.dump()
