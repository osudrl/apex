"""Implementation of vanilla policy gradient."""
import torch
import torch.optim as optim
from torch.autograd import Variable
from baselines.zero_baseline import ZeroBaseline
from utils.math import center
import numpy as np
import copy


class VPG():
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

    def train(self, n_itr, n_trj, max_trj_len, adaptive=True, desired_kl=2e-3):
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

            policy_loss = -(logprobs * advantages + 0.01 * entropy).mean()

            self.optimizer.zero_grad()
            policy_loss.backward()
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

            if np.any(np.isnan(action.data.numpy())):
                print("=================")
                print("mu:")
                print(means.data.numpy())
                print("sigma")
                print(stds.data.numpy())
                print("log sigma")
                print(log_stds.data.numpy())
                print("=================")
                input("paused")

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
