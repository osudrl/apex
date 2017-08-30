"""Proximal Policy Optimization with the clip objective."""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from rl.utils import center, RLDataset
from rl.baselines import ZeroBaseline
from ..base import PolicyGradientAlgorithm

import numpy as np


class PPO(PolicyGradientAlgorithm):
    def __init__(self, env, policy, tau=0.95, discount=0.99, lr=3e-4,
                 baseline=None):
        self.env = env
        self.policy = policy
        self.discount = discount
        self.tau = tau

        if baseline is None:
            self.baseline = ZeroBaseline()
        else:
            self.baseline = baseline

        self.optimizer = optim.Adam(policy.parameters(), lr=lr)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--n_itr", type=int, default=1000,
                            help="number of iterations of the learning algorithm")
        parser.add_argument("--max_trj_len", type=int, default=100,
                            help="maximum trajectory length")
        parser.add_argument("--n_trj", type=int, default=100,
                            help="number of sample trajectories per iteration")
        parser.add_argument("--lr", type=int, default=3e-4,
                            help="Adam learning rate")
        parser.add_argument("--tau", type=int, default=0.95,
                            help="Generalized advantage estimate discount")
        parser.add_argument("--adaptive", type=bool, default=True,
                            help="Adjust learning rate based on kl divergence")

    def train(self, n_itr, n_trj, max_trj_len, epsilon=0.2, epochs=10,
              explore_bonus=0.0, batch_size=64, logger=None):
        env = self.env
        policy = self.policy
        for _ in range(n_itr):
            paths = [self.rollout(env, policy, max_trj_len) for _ in range(n_trj)]

            observations = torch.cat([p["observations"] for p in paths])
            actions = torch.cat([p["actions"] for p in paths])
            advantages = torch.cat([p["advantages"] for p in paths])

            advantages = center(advantages)

            dataset = RLDataset(observations, actions, advantages)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for _ in range(epochs):
                for data in dataloader:
                    obs_batch, action_batch, advantage_batch = map(Variable, data)
                    distribution = policy.get_distribution(obs_batch)
                    old_distribution = distribution.copy()

                    entropy = distribution.entropy()

                    ratio = old_distribution.likelihood_ratio(
                        action_batch,
                        distribution
                    )

                    cpi_loss = ratio * advantage_batch
                    clip_loss = ratio.clamp(1.0 - epsilon, 1.0 + epsilon) \
                                * advantage_batch

                    ppo_loss = -torch.min(cpi_loss, clip_loss).mean()
                    entropy_penalty = -(explore_bonus * entropy).mean()

                    total_loss = ppo_loss + entropy_penalty

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

            self.baseline.fit(paths)

            mean_reward = np.mean(([p["rewards"].sum().data[0] for p in paths]))

            if logger is not None:
                logger.record("Reward", mean_reward)
                logger.dump()
