"""Proximal Policy Optimization with the clip objective."""
from copy import deepcopy

import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.autograd import Variable

from ..base import PolicyGradientAlgorithm

import numpy as np


class PPO(PolicyGradientAlgorithm):
    def __init__(self, env, policy, tau=0.95, discount=0.99, lr=3e-4):
        self.env = env
        self.policy = policy
        self.discount = discount
        self.tau = tau

        self.last_state = None

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
        old_policy = deepcopy(policy)

        for itr in range(n_itr):
            print("********** Iteration %i ************" % itr)
            observations, actions, returns, values, epr = self.sample_steps(env, policy, 2048)
            
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            batch_size = batch_size or advantages.numel()
            print("timesteps in batch: %i" % advantages.size()[0])

            old_policy.load_state_dict(policy.state_dict())  # WAY faster than deepcopy
            if hasattr(policy, 'obs_filter'):
                old_policy.obs_filter = policy.obs_filter

            for _ in range(epochs):
                losses = []
                sampler = BatchSampler(
                    SubsetRandomSampler(range(advantages.numel())),
                    batch_size,
                    drop_last=False
                )

                for indices in sampler:
                    indices = torch.LongTensor(indices)

                    obs_batch = observations[indices]
                    action_batch = actions[indices]

                    return_batch = Variable(returns[indices])
                    advantage_batch = Variable(advantages[indices])

                    values, action_log_probs, _ = policy.evaluate_actions(
                        Variable(obs_batch),
                        Variable(action_batch)
                    )

                    _, old_action_log_probs, _ = old_policy.evaluate_actions(
                        Variable(obs_batch, volatile=True),
                        Variable(action_batch, volatile=True)
                    )

                    old_action_log_probs = Variable(old_action_log_probs.data)

                    ratio = torch.exp(action_log_probs - old_action_log_probs)

                    cpi_loss = ratio * advantage_batch
                    clip_loss = ratio.clamp(1.0 - epsilon, 1.0 + epsilon) * advantage_batch
                    actor_loss = -torch.min(cpi_loss, clip_loss).mean()

                    critic_loss = (return_batch - values).pow(2).mean()

                    self.optimizer.zero_grad()
                    (actor_loss + critic_loss).backward()
                    self.optimizer.step()

                    losses.append([actor_loss.data.clone().numpy()[0],
                                   #entropy_penalty.data.numpy()[0],
                                   critic_loss.data.numpy()[0],
                                   ratio.data.mean()])

                print(' '.join(["%g"%x for x in np.mean(losses, axis=0)]))

            if logger is not None:
                logger.record("Reward", epr)
                logger.dump()
