import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.policies.base import Net, normc_fn

# The base class for a critic. Includes functions for normalizing reward and state (optional)
class Critic(Net):
  def __init__(self):
    super(Critic, self).__init__()

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

class GaussianMLP_Critic(Critic):
  def __init__(self, state_dim, hidden_size=256, hidden_layers=2, env_name='NOT SET', nonlinearity=torch.tanh, init_std=1, learn_std=True, bounded=False, normc_init=True, obs_std=None, obs_mean=None):
    super(GaussianMLP_Critic, self).__init__()

    self.critic_layers = nn.ModuleList()
    self.critic_layers += [nn.Linear(state_dim, hidden_size)]
    for _ in range(hidden_layers-1):
        self.critic_layers += [nn.Linear(hidden_size, hidden_size)]
    self.network_out = nn.Linear(hidden_size, 1)

    self.env_name = env_name
    self.nonlinearity = nonlinearity
    
    self.obs_std = obs_std
    self.obs_mean = obs_mean

    # weight initialization scheme used in PPO paper experiments
    self.normc_init = normc_init
    
    self.init_parameters()
    self.train()

  def init_parameters(self):
    if self.normc_init:
        print("Doing norm column initialization.")
        self.apply(normc_fn)

  def forward(self, inputs):
    if self.training == False:
        inputs = (inputs - self.obs_mean) / self.obs_std

    x = inputs
    for l in self.critic_layers:
        x = self.nonlinearity(l(x))
    value = self.network_out(x)

    return value

  def act(self, inputs):
    value = self(inputs)
    return value


class FF_Critic(Critic):
  def __init__(self, state_dim, action_dim, hidden_size=256, hidden_layers=2, env_name='NOT SET'):
    super(FF_Critic, self).__init__()

    self.critic_layers = nn.ModuleList()
    self.critic_layers += [nn.Linear(state_dim + action_dim, hidden_size)]
    for _ in range(hidden_layers-1):
        self.critic_layers += [nn.Linear(hidden_size, hidden_size)]
    self.network_out = nn.Linear(hidden_size, action_dim)

    self.env_name = env_name
    self.initialize_parameters()

  def forward(self, state, action):

    if len(state.size()) > 2:
      x = torch.cat([state, action], 2)
    elif len(state.size()) > 1:
      x = torch.cat([state, action], 1)
    else:
      x = torch.cat([state, action])

    for idx, layer in enumerate(self.critic_layers):
      x = F.relu(layer(x))

    return self.network_out(x)

class Dual_Q_Critic(Critic):
  def __init__(self, state_dim, action_dim, hidden_size=256, hidden_layers=2, env_name='NOT SET'):
    super(Dual_Q_Critic, self).__init__()

    # Q1 architecture
    self.q1_layers = nn.ModuleList()
    self.q1_layers += [nn.Linear(state_dim + action_dim, hidden_size)]
    for _ in range(hidden_layers-1):
        self.q1_layers += [nn.Linear(hidden_size, hidden_size)]
    self.q1_out = nn.Linear(hidden_size, action_dim)

    # Q2 architecture
    self.q2_layers = nn.ModuleList()
    self.q2_layers += [nn.Linear(state_dim + action_dim, hidden_size)]
    for _ in range(hidden_layers-1):
        self.q2_layers += [nn.Linear(hidden_size, hidden_size)]
    self.q2_out = nn.Linear(hidden_size, action_dim)

    self.env_name = env_name

  def forward(self, state, action):

    # print(state.size(), state)
    #print(action.size(), action)
    if len(state.size()) > 2:
      x1 = torch.cat([state, action], 2)
    elif len(state.size()) > 1:
      x1 = torch.cat([state, action], 1)
    else:
      x1 = torch.cat([state, action])

    x2 = x1

    # Q1 forward
    for idx, layer in enumerate(self.q1_layers):
      x1 = F.relu(layer(x1))

    # Q2 forward
    for idx, layer in enumerate(self.q2_layers):
      x2 = F.relu(layer(x2))

    return self.q1_out(x1), self.q2_out(x2)

  def Q1(self, state, action):
    #print(state.size(), state)
    #print(action.size(), action)
    if len(state.size()) > 2:
      x1 = torch.cat([state, action], 2)
    elif len(state.size()) > 1:
      x1 = torch.cat([state, action], 1)
    else:
      x1 = torch.cat([state, action])
    
    # Q1 forward
    for idx, layer in enumerate(self.q1_layers):
      x1 = F.relu(layer(x1))

    return self.q1_out(x1)


class LSTM_Critic(Critic):
  def __init__(self, input_dim, action_dim, hidden_size=64, hidden_layers=1, env_name='NOT SET'):
    super(LSTM_Critic, self).__init__()

    self.critic_layers = nn.ModuleList()
    self.critic_layers += [nn.LSTMCell(input_dim + action_dim, hidden_size)]
    for _ in range(hidden_layers-1):
        self.critic_layers += [nn.LSTMCell(hidden_size, hidden_size)]
    self.network_out = nn.Linear(hidden_size, action_dim)

    self.init_hidden_state()

    self.is_recurrent = True
    self.env_name = env_name

  def get_hidden_state(self):
    return self.hidden, self.cells

  def init_hidden_state(self, batch_size=1):
    self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.critic_layers]
    self.cells  = [torch.zeros(batch_size, l.hidden_size) for l in self.critic_layers]
  
  def forward(self, state, action):
    if len(state.size()) != len(action.size()):
      print("state and action must have same number of dimensions: {} vs {}", state.size(), action.size())
      exit(1)

    if len(state.size()) == 3: # if we get a batch of trajectories
      self.init_hidden_state(batch_size=state.size(1))
      value = []
      for t, (state_batch_t, action_batch_t) in enumerate(zip(state, action)):
        #print(state_batch_t.size(), action_batch_t.size())
        x_t = torch.cat([state_batch_t, action_batch_t], 1)

        for idx, layer in enumerate(self.critic_layers):
          c, h = self.cells[idx], self.hidden[idx]
          self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
          x_t = self.hidden[idx]
        x_t = self.network_out(x_t)
        value.append(x_t)

      x = torch.stack([a.float() for a in value])

    elif len(state.size()) == 2: # if we get a trajectory
      self.init_hidden_state()

      value = []
      for t, (state_t, action_t) in enumerate(zip(state, action)):
        x_t = torch.cat([state_t, action_t])
        x_t = x_t.view(1, -1)

        for idx, layer in enumerate(self.critic_layers):
          c, h = self.cells[idx], self.hidden[idx]
          #self.cells[idx], self.hidden[idx] = layer(x_t, (c, h))
          self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
          x_t = self.hidden[idx]
        x_t = self.network_out(x_t)
        value.append(x_t)

      x = torch.cat([a.float() for a in value])

    elif len(state.size()) == 1: # if we get a single timestep
      print("MAKE SURE THIS WORKS.")
      x = torch.cat([state_t, action_t], 1)
      x = x.view(1, -1)

      for idx, layer in enumerate(self.critic_layers):
        c, h = self.cells[idx], self.hidden[idx]
        #self.cells[idx], self.hidden[idx] = layer(x, (c, h))
        self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
        x = self.hidden[idx]
      x = self.network_out(x)

    else:
      print("Invalid input dimensions.")
      exit(1)
    return x

