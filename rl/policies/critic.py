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
      if len(r.size()) == 1:
        r_old = self.welford_reward_mean
        self.welford_reward_mean += (r - r_old) / self.welford_reward_n
        self.welford_reward_mean_diff += (r - r_old) * (r - r_old)
        self.welford_reward_n += 1
      elif len(r.size()) == 2:
        for r_n in r:
          r_old = self.welford_reward_mean
          self.welford_reward_mean += (r_n - r_old) / self.welford_reward_n
          self.welford_reward_mean_diff += (r_n - r_old) * (r_n - r_old)
          self.welford_reward_n += 1
      else:
        raise NotImplementedError

    return (r - self.welford_reward_mean) / torch.sqrt(self.welford_reward_mean_diff / self.welford_reward_n)

  def normalize_state(self, state, update=True):
    if update:
      state_old = self.welford_state_mean
      self.welford_state_mean += (state - state_old) / self.welford_state_n
      self.welford_state_mean_diff += (state - state_old) * (state - state_old)
      self.welford_state_n += 1
    return (state - self.welford_state_mean) / torch.sqrt(self.welford_state_mean_diff / self.welford_state_n)


class FF_Critic(Critic):
  def __init__(self, state_dim, action_dim, hidden_size=256, hidden_layers=2, env_name='NOT SET'):
    super(FF_Critic, self).__init__()
    #self.l1 = nn.Linear(state_dim + action_dim, hidden_size)
    #self.l2 = nn.Linear(hidden_size, hidden_size)
    #self.l3 = nn.Linear(hidden_size, 1)

    self.critic_layers = nn.ModuleList()
    self.critic_layers += [nn.Linear(state_dim+action_dim, hidden_size)]
    for _ in range(hidden_layers-1):
        self.critic_layers += [nn.Linear(hidden_size, hidden_size)]
    self.network_out = nn.Linear(hidden_size, action_dim)

    self.env_name = env_name

  def forward(self, state, action):
    #x = F.relu(self.l1(torch.cat([state, action], 1)))
    #x = F.relu(self.l2(x))
    #return self.l3(x)

    x = torch.cat([state, action], 1)
    for idx, layer in enumerate(self.critic_layers):
      x = F.relu(layer(x))

    return self.network_out(x)

class LSTM_Critic(Critic):
  def __init__(self, input_dim, action_dim, hidden_size=32, hidden_layers=1):
    super(Recurrent_Critic, self).__init__()

    self.actor_layers = nn.ModuleList()
    self.actor_layers += [nn.LSTMCell(input_dim, hidden_size)]
    for _ in range(hidden_layers-1):
        self.actor_layers += [nn.LSTMCell(hidden_size, hidden_size)]
    self.network_out = nn.Linear(hidden_size, action_dim)

    self.init_hidden_state()

    self.recurrent = True

  def get_hidden_state(self):
    return self.hidden, self.cells

  def init_hidden_state(self):
    self.hidden = [torch.zeros(1, l.hidden_size) for l in self.actor_layers]
    self.cells  = [torch.zeros(1, l.hidden_size) for l in self.actor_layers]
  
  def forward(self, x):
    print("MAKE SURE THIS WORKS.")
    if len(x.size()) == 1:
      x = x.view(1, -1)

    for idx, layer in enumerate(self.actor_layers):
      c, h = self.cells[idx], self.hidden[idx]
      self.cells[idx], self.hidden[idx] = layer(x, (c, h))
      x = self.hidden[idx]
    x = self.network_out(x)
    return torch.tanh(x)
