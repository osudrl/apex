import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math


def lognormal_density(x, mean, log_std, std):
    var = std.pow(2)

    log_density = -(x - mean).pow(2) / (
        2 * var) - 0.5 * math.log(2 * math.pi) - log_std

    return log_density.sum(1)


class VPG():
    def __init__(self, env, policy, discount=0.99):
        self.env = env
        self.policy = policy
        #self.baseline = baseline
        self.discount = discount

        self.optimizer = optim.Adam(policy.parameters(), lr=0.01)

    def train(self, n_itr, n_trj, max_trj_len):
        env = self.env
        policy = self.policy
        for _ in range(n_itr):

            losses = []
            total_r = []
            for _ in range(n_trj):
                rewards = []

                obs = env.reset()

                logprobs = []
                for _ in range(max_trj_len):
                    obs_var = Variable(torch.Tensor(obs).unsqueeze(0))
                    means, log_stds, stds = policy(obs_var)

                    action = torch.normal(means, stds)
                    action = action.detach()

                    logprobs.append(
                        lognormal_density(action, means, log_stds, stds)
                    )

                    next_obs, reward, done, _ = env.step(action.data.numpy())
                    rewards.append(reward)

                    obs = next_obs

                    if done:
                        break

                total_r.append(sum(rewards))

                R = Variable(torch.zeros(1, 1))
                loss = 0
                for i in reversed(range(len(rewards))):
                    R = self.discount * R + rewards[i]
                    loss = loss - logprobs[i] * R

                losses.append(loss)

            print(sum(total_r) / len(total_r))

            policy_loss = sum(losses) / len(losses)

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
