import torch
import torch.nn as nn
import torch.nn.functional as F

# The base class for an actor. Includes functions for normalizing state (optional)
class Actor(nn.Module):
  def __init__(self):
    super(Actor, self).__init__()
    self.is_recurrent = False

    self.welford_state_mean = 0.0
    self.welford_state_mean_diff = 1.0
    self.welford_state_n = 1

  def forward(self):
    raise NotImplementedError
  
  def normalize_state(self, state, update=True):
    if update:
      state_old = self.welford_state_mean
      self.welford_state_mean += (state - state_old) / self.welford_state_n
      self.welford_state_mean_diff += (state - state_old) * (state - state_old)
      self.welford_state_n += 1
    return (state - self.welford_state_mean) / sqrt(self.welford_state_mean_diff / self.welford_state_n)


class FF_Actor(Actor):
  def __init__(self, state_dim, action_dim, hidden_size=256):
    super(FF_Actor, self).__init__()
    self.l1 = nn.Linear(state_dim, hidden_size)
    self.l2 = nn.Linear(hidden_size, hidden_size)
    self.l3 = nn.Linear(hidden_size, action_dim)

  def forward(self, state):
    x = F.relu(self.l1(state))
    x = F.relu(self.l2(x))
    return torch.tanh(self.l3(x))
