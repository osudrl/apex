import torch
import torch.nn as nn
import torch.nn.functional as F

# The base class for a critic. Includes functions for normalizing reward and state (optional)
class Critic(nn.Module):
  def __init__(self):
    super(Critic, self).__init__()
    self.is_recurrent = False

    self.welford_state_mean = 0.0
    self.welford_state_mean_diff = 1.0
    self.welford_state_n = 1

    self.welford_reward_mean = 0.0
    self.welford_reward_mean_diff = 1.0
    self.welford_reward_n = 1

  def forward(self):
    raise NotImplementedError
  
  def normalize_reward(self, r, update=True):
    if update:
      r_old = self.welford_reward_mean
      self.welford_reward_mean += (r - r_old) / self.welford_reward_n
      self.welford_reward_mean_diff += (r - r_old) * (r - r_old)
      self.welford_reward_n += 1

    return (r - self.welford_reward_mean) / sqrt(self.welford_reward_mean_diff / self.welford_reward_n)

  def normalize_state(self, state, update=True):
    if update:
      state_old = self.welford_state_mean
      self.welford_state_mean += (state - state_old) / self.welford_state_n
      self.welford_state_mean_diff += (state - state_old) * (state - state_old)
      self.welford_state_n += 1
    return (state - self.welford_state_mean) / sqrt(self.welford_state_mean_diff / self.welford_state_n)


class DDPG_Critic(Critic):
  def __init__(self, state_dim, action_dim, hidden_size=256):
    super(DDPG_Critic, self).__init__()
    self.l1 = nn.Linear(state_dim + action_dim, hidden_size)
    self.l2 = nn.Linear(hidden_size, hidden_size)
    self.l3 = nn.Linear(hidden_size, 1)

  def forward(self, state, action):
    x = F.relu(self.l1(torch.cat([state, action], 1)))
    x = F.relu(self.l2(x))
    return self.l3(x)
