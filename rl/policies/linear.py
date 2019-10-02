import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearMLP(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_size=32):
    super(LinearMLP, self).__init__()

    self.l1 = nn.Linear(state_dim, hidden_size)
    self.l2 = nn.Linear(hidden_size, action_dim)

    for p in self.parameters():
      p.data = torch.zeros(p.shape)

    self.normalizer_mean = torch.zeros(state_dim, requires_grad=False)
    self.normalizer_mean_diff = torch.ones(state_dim, requires_grad=False)
    self.n = 1
  
  def forward(self, state, update_normalizer=False):
    if update_normalizer:
        self.update_normalization_stats(state)
      
    state = (state - self.normalizer_mean) / torch.sqrt(self.normalizer_mean_diff/self.n) # center state according to observed state distribution

    a = self.l1(state)
    return self.l2(a)
  
  def get_normalization_stats(self):
    return self.n, self.normalizer_mean, self.normalizer_mean_diff

  def set_normalization_stats(self, n, mean, mean_diff):
    self.n = n
    self.normalizer_mean = mean
    self.normalizer_mean_diff = mean_diff

  def update_normalization_stats(self, state):
    old_mean = self.normalizer_mean

    self.normalizer_mean += (state - old_mean)/self.n
    self.normalizer_mean_diff += (state - old_mean) * (state - self.normalizer_mean)
    self.n += 1
