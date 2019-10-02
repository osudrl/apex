import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentNet(nn.Module):
  def __init__(self, input_dim, action_dim, hidden_size=6, hidden_layers=3, has_critic=False):
    super(RecurrentActor, self).__init__()

    self.actor_layers = nn.ModuleList()
    self.actor_layers += [nn.LSTMCell(input_dim, hidden_size)]
    for _ in range(hidden_layers-1):
        self.actor_layers += [nn.LSTMCell(hidden_size, hidden_size)]

    self.network_out = nn.Linear(hidden_size, action_dim)

    #ref: https://discuss.pytorch.org/t/correct-way-to-declare-hidden-and-cell-states-of-lstm/15745/3
    #possibly outdated? tensors might be good enough to save hidden states, may not need nn.Variable()

    self.hidden = [torch.zeros(1, hidden_size) for _ in self.actor_layers]
    self.cells = [torch.zeros(1, hidden_size) for _ in self.actor_layers]

  def get_hidden_state(self):
    return self.hidden, self.cells

  def init_hidden_state(self):
    self.hidden = [torch.zeros(1, hidden_size) for _ in self.actor_layers]
    self.cells = [torch.zeros(1, hidden_size) for _ in self.actor_layers]
  
  def forward(self, x):
    for idx, layer in enumerate(self.actor_layers): #, self.hidden, self.cells):
      c, h = self.cells[idx], self.hidden[idx]
      self.cells[idx], self.hidden[idx] = layer(x, (c, h))
      x = self.hidden[idx]

    x = self.network_out(x)
    return F.tanh(x)
