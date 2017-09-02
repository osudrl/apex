from torch.autograd import Variable
import torch

import numpy as np


class PolicyGradientAlgorithm():
    def train():
        raise NotImplementedError

    @staticmethod
    def add_arguments(parser):
        raise NotImplementedError

    def rollout(self, env, policy, max_trj_len):
        """Collect a single rollout."""

        observations = []
        actions = []
        rewards = []
        returns = []
        advantages = []
        values = []

        obs = env.reset().ravel()[None, :]
        obs_var = Variable(torch.Tensor(obs))

        values.append(policy.get_critic(obs_var))
        for _ in range(max_trj_len):
            action = policy.get_action(obs_var)

            next_obs, reward, done, _ = env.step(action.data.numpy().ravel())

            observations.append(obs_var)
            actions.append(action)
            rewards.append(Variable(torch.Tensor([[reward]])))

            obs = next_obs.ravel()[None, :]
            obs_var = Variable(torch.Tensor(obs))

            values.append(policy.get_critic(obs_var))

            if done:
                break

        R = Variable(torch.zeros(1, 1))
        advantage = Variable(torch.zeros(1, 1))
        for t in reversed(range(len(rewards))):
            R = self.discount * R + rewards[t]

            # generalized advantage estimation
            # see: https://arxiv.org/abs/1506.02438
            delta = rewards[t] + self.discount * values[t + 1] - values[t]
            advantage = advantage * self.discount * self.tau + delta

            returns.append(R)
            advantages.append(advantage)

        return dict(
            rewards=torch.stack(rewards),
            returns=torch.cat(returns[::-1]),
            advantages=torch.cat(advantages[::-1]),
            observations=torch.cat(observations),
            actions=torch.cat(actions)
        )
