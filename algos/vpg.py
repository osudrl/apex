"""Implementation of vanilla policy gradient."""
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


class VPG():
    """
    Implements vanilla policy gradient aka REINFORCE.

    http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf
    """

    def __init__(self, env, policy, discount=0.99, lr=0.01):
        self.env = env
        self.policy = policy
        #self.baseline = baseline
        self.discount = discount

        self.optimizer = optim.Adam(policy.parameters(), lr=lr)

    def train(self, n_itr, n_trj, max_trj_len):
        env = self.env
        policy = self.policy
        for _ in range(n_itr):
            paths = [self.rollout(env, policy, max_trj_len) for _ in range(n_trj)]

            observations = torch.cat([p["observations"] for p in paths])
            actions = torch.cat([p["actions"] for p in paths])
            returns = torch.cat([p["returns"] for p in paths])

            means, log_stds, stds = policy(observations)

            logprobs = policy.log_likelihood(actions, means, log_stds, stds)

            policy_loss = -(logprobs * returns).mean()

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            print(
                'Average Return: %f' %
                np.mean(([p["rewards"].sum().data.numpy() for p in paths]))
            )

    def rollout(self, env, policy, max_trj_len):
        """Collect a single rollout."""
        rewards = []
        observations = []
        actions = []
        returns = []

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

        R = Variable(torch.zeros(1, 1))
        for i in reversed(range(len(rewards))):
            R = self.discount * R + rewards[i]
            returns.append(R)

        return dict(
            rewards=torch.stack(rewards),
            returns=torch.cat(returns[::-1]),
            observations=torch.cat(observations),
            actions=torch.cat(actions)
        )
