import time
import numpy as np
import os
import torch

class ARS:
  def __init__(self, policy, env, step_size=0.02, std=0.0075, deltas=16, top_n=None, reward_shift=0):
    self.std = std
    self.num_deltas = deltas
    self.param_shape = [x.shape for x in policy.parameters()]
    self.policy = policy
    self.env = env
    self.step_size = step_size

    self.reward_shift = reward_shift
    if top_n is None:
      self.top_n = self.num_deltas
    else:
      self.top_n = top_n

    for p in self.policy.parameters():
      p.data = torch.zeros(p.shape)

  def generate_delta(self):
    delta = [torch.normal(0, self.std, shape) for shape in self.param_shape]
    return delta

  def step(self, black_box):
    deltas = []
    rewards = []
    for n in range(self.num_deltas):
      d = self.generate_delta()

      for p, dp in zip(self.policy.parameters(), d):
        p.data += dp;
      r_pos = black_box(self.policy)

      for p, dp in zip(self.policy.parameters(), d):
        p.data -= 2*dp;
      r_neg = black_box(self.policy)

      for p, dp in zip(self.policy.parameters(), d):
        p.data += dp;

      deltas.append([d, r_pos, r_neg])

      rewards.append(r_pos)
      rewards.append(r_neg)
    
    reward_std = np.std(rewards)
    factor = self.step_size / (self.top_n * reward_std)
    for d, r_pos, r_neg in deltas:
      for p, dp in zip(self.policy.parameters(), d):
        p.data += factor * (r_pos - r_neg) * dp

