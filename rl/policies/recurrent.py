import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentNet(nn.Module):
  def __init__(self, input_dim, action_dim, hidden_size=32, hidden_layers=1, has_critic=False):
    super(RecurrentNet, self).__init__()

    self.actor_layers = nn.ModuleList()
    self.actor_layers += [nn.LSTMCell(input_dim, hidden_size)]
    for _ in range(hidden_layers-1):
        self.actor_layers += [nn.LSTMCell(hidden_size, hidden_size)]

    self.network_out = nn.Linear(hidden_size, action_dim)

    self.init_hidden_state()

    self.normalizer_mean = torch.zeros(input_dim, requires_grad=False)
    self.normalizer_mean_diff = torch.ones(input_dim, requires_grad=False)
    self.n = 1

  def get_hidden_state(self):
    return self.hidden, self.cells

  def init_hidden_state(self):
    self.hidden = [torch.zeros(1, l.hidden_size) for l in self.actor_layers]
    self.cells  = [torch.zeros(1, l.hidden_size) for l in self.actor_layers]
  
  def forward(self, x, update_normalizer=False):
    if update_normalizer:
        self.update_normalization_stats(x)
      
    # the below may not work for batch tensors
    x = (x - self.normalizer_mean) / torch.sqrt(self.normalizer_mean_diff/self.n) # center state according to observed state distribution

    if len(x.size()) == 1:
      x = x.view(1, -1)

    for idx, layer in enumerate(self.actor_layers):
      c, h = self.cells[idx], self.hidden[idx]
      self.cells[idx], self.hidden[idx] = layer(x, (c, h))
      x = self.hidden[idx]

    x = self.network_out(x)
    return torch.tanh(x)

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
