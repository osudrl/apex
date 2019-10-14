import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import sqrt

# The base class for an actor. Includes functions for normalizing state (optional)
class Actor(nn.Module):
  def __init__(self):
    super(Actor, self).__init__()
    self.is_recurrent = False

    self.welford_state_mean = 0.0
    self.welford_state_mean_diff = 1.0
    self.welford_state_n = 1

    self.env_name = None

  def forward(self):
    raise NotImplementedError

  def get_action(self):
    raise NotImplementedError
  
  def normalize_state(self, state, update=True):
    if update:
      if len(state.size()) == 1:
        state_old = self.welford_state_mean
        self.welford_state_mean += (state - state_old) / self.welford_state_n
        self.welford_state_mean_diff += (state - state_old) * (state - state_old)
        self.welford_state_n += 1
      elif len(state.size()) == 2:
        for r_n in r:
          state_old = self.welford_state_mean
          self.welford_state_mean += (state_n - state_old) / self.welford_state_n
          self.welford_state_mean_diff += (state_n - state_old) * (state_n - state_old)
          self.welford_state_n += 1
      else:
        raise NotImplementedError
    return (state - self.welford_state_mean) / sqrt(self.welford_state_mean_diff / self.welford_state_n)

class Linear_Actor(Actor):
  def __init__(self, state_dim, action_dim, hidden_size=32):
    super(Linear_Actor, self).__init__()

    self.l1 = nn.Linear(state_dim, hidden_size)
    self.l2 = nn.Linear(hidden_size, action_dim)

    for p in self.parameters():
      p.data = torch.zeros(p.shape)

  def forward(self, state):
    a = self.l1(state)
    a = self.l2(a)
    self.action = a
    return a

class FF_Actor(Actor):
  def __init__(self, state_dim, action_dim, hidden_size=256, hidden_layers=2, env_name='NOT SET', nonlinearity=F.relu):
    super(FF_Actor, self).__init__()

    self.actor_layers = nn.ModuleList()
    self.actor_layers += [nn.Linear(state_dim, hidden_size)]
    for _ in range(hidden_layers-1):
        self.actor_layers += [nn.Linear(hidden_size, hidden_size)]
    self.network_out = nn.Linear(hidden_size, action_dim)

    self.action = None
    self.env_name = env_name
    self.nonlinearity = nonlinearity

  def forward(self, state):
    x = state
    for idx, layer in enumerate(self.actor_layers):
      x = self.nonlinearity(layer(x))

    self.action = torch.tanh(self.network_out(x))
    return self.action

  def get_action(self):
    return self.action

class LSTM_Actor(Actor):
  def __init__(self, input_dim, action_dim, hidden_size=64, hidden_layers=1, env_name='NOT SET', nonlinearity=torch.tanh):
    super(LSTM_Actor, self).__init__()

    self.actor_layers = nn.ModuleList()
    self.actor_layers += [nn.LSTMCell(input_dim, hidden_size)]
    for _ in range(hidden_layers-1):
        self.actor_layers += [nn.LSTMCell(hidden_size, hidden_size)]
    self.network_out = nn.Linear(hidden_size, action_dim)

    self.action = None
    self.init_hidden_state()
    self.env_name = env_name
    self.nonlinearity = nonlinearity
    
    self.recurrent = True

  def get_hidden_state(self):
    return self.hidden, self.cells

  def init_hidden_state(self):
    self.hidden = [torch.zeros(1, l.hidden_size) for l in self.actor_layers]
    self.cells  = [torch.zeros(1, l.hidden_size) for l in self.actor_layers]

  def forward(self, x):
    #print("MAKE SURE THIS WORKS.")
    if len(x.size()) == 1:
      x = x.view(1, -1)

    for idx, layer in enumerate(self.actor_layers):
      c, h = self.cells[idx], self.hidden[idx]
      self.cells[idx], self.hidden[idx] = layer(x, (c, h))
      x = self.hidden[idx]
    x = self.nonlinearity(self.network_out(x))
    self.action = x
    return x
  
  def get_action(self):
    return self.action

