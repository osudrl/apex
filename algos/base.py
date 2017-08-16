from torch.autograd import Variable
import torch

import numpy as np


class PolicyGradientAlgorithm():
    def train():
        raise NotImplementedError

    def rollout(self, env, policy, max_trj_len):
        """Collect a single rollout."""

        observations = []
        actions = []
        rewards = []
        returns = []
        advantages = []

        obs = env.reset().ravel()[None, :]
        for _ in range(max_trj_len):
            obs_var = Variable(torch.Tensor(obs))

            action = policy.get_action(obs_var)

            next_obs, reward, done, _ = env.step(action.data.numpy().ravel())

            observations.append(obs_var)
            actions.append(action)
            rewards.append(Variable(torch.Tensor([reward])))

            obs = next_obs.ravel()[None, :]

            if done:
                break

        baseline = Variable(self.baseline.predict(torch.cat(observations)))
        R = Variable(torch.zeros(1, 1))
        for t in reversed(range(len(rewards))):
            R = self.discount * R + rewards[t]
            advantage = R - baseline[t]

            returns.append(R)
            advantages.append(advantage)

        return dict(
            rewards=torch.stack(rewards),
            returns=torch.cat(returns[::-1]),
            advantages=torch.cat(advantages[::-1]),
            observations=torch.cat(observations),
            actions=torch.cat(actions)
        )
