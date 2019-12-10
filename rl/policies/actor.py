import torch
import torch.nn as nn
import torch.nn.functional as F

#from rl.distributions import DiagonalGaussian

from torch import sqrt

from rl.policies.base import Net

class Actor(Net):
  def __init__(self):
    super(Actor, self).__init__()

  def forward(self):
    raise NotImplementedError

  def get_action(self):
    raise NotImplementedError

class Linear_Actor(Actor):
  def __init__(self, state_dim, action_dim, hidden_size=32):
    super(Linear_Actor, self).__init__()

    self.l1 = nn.Linear(state_dim, hidden_size)
    self.l2 = nn.Linear(hidden_size, action_dim)

    self.action_dim = action_dim

    for p in self.parameters():
      p.data = torch.zeros(p.shape)

  def forward(self, state):
    a = self.l1(state)
    a = self.l2(a)
    self.action = a
    return a

  def get_action(self):
    return self.action

# Actor network for gaussian mlp
#class GaussianMLP_Actor(Actor):
class Gaussian_FF_Actor(Actor): # more consistent with other actor naming conventions
  def __init__(self, state_dim, action_dim, layers=(256, 256), env_name='NOT SET', nonlinearity=torch.tanh, init_std=1, learn_std=True, bounded=False, normc_init=True, obs_std=None, obs_mean=None):
    super(Gaussian_FF_Actor, self).__init__()

    self.actor_layers = nn.ModuleList()
    self.actor_layers += [nn.Linear(state_dim, layers[0])]
    for i in range(len(layers)-1):
        self.actor_layers += [nn.Linear(layers[i], layers[i+1])]
    self.means = nn.Linear(layers[-1], action_dim)

    if learn_std == True: # probably don't want to use this for ppo, always use fixed std
      self.log_stds = nn.Linear(layers[-1], action_dim)
      self.learn_std = True
    else:
      self.fixed_std = init_std
      self.learn_std = False

    self.action = None
    self.action_dim = action_dim
    self.env_name = env_name
    self.nonlinearity = nonlinearity

    self.obs_std = obs_std
    self.obs_mean = obs_mean

    # weight initialization scheme used in PPO paper experiments
    self.normc_init = normc_init

    self.bounded = bounded

    self.init_parameters()
    self.train()

  def init_parameters(self):
    if self.normc_init:
        self.apply(normc_fn)
        self.means.weight.data.mul_(0.01)

  def _get_dist_params(self, state):
    if self.training == False:
        state = (state - self.obs_mean) / self.obs_std

    x = state
    for l in self.actor_layers:
        x = self.nonlinearity(l(x))
    x = self.means(x)

    if self.bounded:
        mean = torch.tanh(x) 
    else:
        mean = x

    if self.learn_std:
      sd = torch.clamp(self.log_stds(x), -20, 2).exp() # TODO: make these a constant or something
    else:
      sd = self.fixed_std

    return mean, sd

  def forward(self, inputs):
    mean, _ = self._get_dist_params(inputs)
    self.action = mean

    return mean

  def get_action(self):
    return self.action

  def act(self, inputs, deterministic=True): # make true by default for evaluation purposes
    mu, sd = self._get_dist_params(inputs)
    if not deterministic:
      self.action = torch.distributions.Normal(mu, sd).sample()
    else:
      self.action = mu

    return self.action.detach()

  def evaluate(self, inputs):
    mu, sd = self._get_dist_params(inputs)
    return torch.distributions.Normal(mu, sd)

class FF_Actor(Actor):
  def __init__(self, state_dim, action_dim, layers=(256, 256), env_name='NOT SET', nonlinearity=F.relu, max_action=1):
    super(FF_Actor, self).__init__()

    self.actor_layers = nn.ModuleList()
    self.actor_layers += [nn.Linear(state_dim, layers[0])]
    for i in range(len(layers)-1):
        self.actor_layers += [nn.Linear(layers[i], layers[i+1])]
    self.network_out = nn.Linear(layers[-1], action_dim)

    self.action = None
    self.action_dim = action_dim
    self.env_name = env_name
    self.nonlinearity = nonlinearity

    self.initialize_parameters()

    self.max_action = max_action

  def forward(self, state):
    x = state
    for idx, layer in enumerate(self.actor_layers):
      x = self.nonlinearity(layer(x))

    self.action = torch.tanh(self.network_out(x))
    return self.action * self.max_action

  def get_action(self):
    return self.action

class LSTM_Actor(Actor):
  def __init__(self, input_dim, action_dim, layers=(128, 128), env_name='NOT SET', nonlinearity=torch.tanh, max_action=1):
    super(LSTM_Actor, self).__init__()

    self.actor_layers = nn.ModuleList()
    self.actor_layers += [nn.LSTMCell(state_dim, layers[0])]
    for i in range(len(layers)-1):
        self.actor_layers += [nn.LSTMCell(layers[i], layers[i+1])]
    self.network_out = nn.Linear(layers[i-1], action_dim)

    self.action = None
    self.action_dim = action_dim
    self.init_hidden_state()
    self.env_name = env_name
    self.nonlinearity = nonlinearity
    
    self.is_recurrent = True

    self.max_action = max_action

  def get_hidden_state(self):
    return self.hidden, self.cells

  def set_hidden_state(self, data):
    if len(data) != 2:
      print("Got invalid hidden state data.")
      exit(1)

    self.hidden, self.cells = data
    
  def init_hidden_state(self, batch_size=1):
    self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]
    self.cells  = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]

  def forward(self, x):
    dims = len(x.size())

    if dims == 3: # if we get a batch of trajectories
      self.init_hidden_state(batch_size=x.size(1))
      action = []
      for t, x_t in enumerate(x):

        for idx, layer in enumerate(self.actor_layers):
          c, h = self.cells[idx], self.hidden[idx]
          self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
          x_t = self.hidden[idx]
        x_t = self.nonlinearity(self.network_out(x_t))
        action.append(x_t)

      x = torch.stack([a.float() for a in action])
      self.action = x

    else:
      if dims == 1: # if we get a single timestep (if not, assume we got a batch of single timesteps)
        x = x.view(1, -1)

      for idx, layer in enumerate(self.actor_layers):
        h, c = self.hidden[idx], self.cells[idx]
        self.hidden[idx], self.cells[idx] = layer(x, (h, c))
        x = self.hidden[idx]
      x = self.nonlinearity(self.network_out(x))

      if dims == 1:
        x = x.view(-1)
      self.action = x

    return x
  
  def get_action(self):
    return self.action

class Gaussian_LSTM_Actor(Actor):
  def __init__(self, state_dim, action_dim, layers=(128, 128), env_name=None, nonlinearity=F.tanh, normc_init=False, max_action=1, fixed_std=None):
    super(Gaussian_LSTM_Actor, self).__init__()

    self.actor_layers = nn.ModuleList()
    self.actor_layers += [nn.LSTMCell(state_dim, layers[0])]
    for i in range(len(layers)-1):
        self.actor_layers += [nn.LSTMCell(layers[i], layers[i+1])]
    self.network_out = nn.Linear(layers[i-1], action_dim)

    self.action = None
    self.action_dim = action_dim
    self.init_hidden_state()
    self.env_name = env_name
    self.nonlinearity = nonlinearity
    self.max_action = max_action
    
    self.is_recurrent = True

    if fixed_std is None:
      self.log_stds = nn.Linear(layers[-1], action_dim)
      self.learn_std = True
    else:
      self.fixed_std = fixed_std
      self.learn_std = False

    if normc_init:
      self.initialize_parameters()

  def _get_dist_params(self, state):
    dims = len(state.size())

    x = state
    if dims == 3: # if we get a batch of trajectories
      self.init_hidden_state(batch_size=x.size(1))
      action = []
      for t, x_t in enumerate(x):

        for idx, layer in enumerate(self.actor_layers):
          c, h = self.cells[idx], self.hidden[idx]
          self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
          x_t = self.hidden[idx]
        #x_t = self.nonlinearity(self.network_out(x_t))
        x_t = self.network_out(x_t)
        action.append(x_t)

      x = torch.stack([a.float() for a in action])

    else:
      if dims == 1: # if we get a single timestep (if not, assume we got a batch of single timesteps)
        x = x.view(1, -1)

      for idx, layer in enumerate(self.actor_layers):
        h, c = self.hidden[idx], self.cells[idx]
        self.hidden[idx], self.cells[idx] = layer(x, (h, c))
        x = self.hidden[idx]
      #x = self.nonlinearity(self.network_out(x))[0]
      x = self.network_out(x)

      if dims == 1:
        x = x.view(-1)

    mu = x
    if self.learn_std:
      sd = torch.clamp(self.log_stds(x), -20, 2).exp()
    else:
      sd = self.fixed_std

    return mu, sd

  def init_hidden_state(self, batch_size=1):
    self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]
    self.cells  = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]

  def forward(self, state, deterministic=True):
    mu, sd = self._get_dist_params(state)

    if not deterministic:
      self.action = torch.distributions.Normal(mu, sd).sample()
    else:
      self.action = mu

    return self.action

  def pdf(self, state):
    mu, sd = self._get_dist_params(state)
    return torch.distributions.Normal(mu, sd)

  def get_action(self):
    return self.action

## Initialization scheme for gaussian mlp (from ppo paper)
# NOTE: the fact that this has the same name as a parameter caused a NASTY bug
# apparently "if <function_name>" evaluates to True in python...
def normc_fn(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

GaussianMLP_Actor = Gaussian_FF_Actor # for legacy code compatibility
