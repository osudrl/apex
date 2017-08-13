"""Implementation of vanilla policy gradient."""
import torch
import torch.optim as optim
from torch.autograd import Variable
from baselines.zero_baseline import ZeroBaseline
import numpy as np


class VPG():
    """
    Implements vanilla policy gradient aka REINFORCE.

    http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf
    """

    def __init__(self, env, policy, discount=0.99, lr=0.01,
                 baseline=None):
        self.env = env
        self.policy = policy
        self.discount = discount

        if baseline is None:
            self.baseline = ZeroBaseline()
        else:
            self.baseline = baseline

        self.optimizer = optim.Adam(policy.parameters(), lr=lr)

    def train(self, n_itr, n_trj, max_trj_len):
        env = self.env
        policy = self.policy
        for _ in range(n_itr):
            paths = [self.rollout(env, policy, max_trj_len) for _ in range(n_trj)]

            observations = torch.cat([p["observations"] for p in paths])
            actions = torch.cat([p["actions"] for p in paths])
            advantages = torch.cat([p["advantages"] for p in paths])

            means, log_stds, stds = policy(observations)

            logprobs = policy.log_likelihood(actions, means, log_stds, stds)

            policy_loss = -(logprobs * advantages).mean()

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            self.baseline.fit(paths)

            print(
                'Average Return: %f' %
                np.mean(([p["rewards"].sum().data.numpy() for p in paths]))
            )

    def rollout(self, env, policy, max_trj_len):
        """Collect a single rollout."""
        observations = []
        actions = []
        rewards = []
        returns = []
        advantages = []

        obs = env.reset()
        for _ in range(max_trj_len):
            obs_var = Variable(torch.Tensor(obs).unsqueeze(0))

            means, log_stds, stds = policy(obs_var)
            action = policy.get_action(means, stds)

            next_obs, reward, done, _ = env.step(action.data.numpy())

            observations.append(obs_var)
            actions.append(action)
            rewards.append(Variable(torch.Tensor([reward])))

            obs = next_obs

            if done:
                break

        baseline = Variable(self.baseline.predict(torch.cat(observations)))
        R = Variable(torch.zeros(1, 1))
        for i in reversed(range(len(rewards))):
            R = self.discount * R + rewards[i]
            advantage = R - baseline[i]

            returns.append(R)
            advantages.append(advantage)

        return dict(
            rewards=torch.stack(rewards),
            returns=torch.cat(returns[::-1]),
            advantages=torch.cat(advantages[::-1]),
            observations=torch.cat(observations),
            actions=torch.cat(actions)
        )
