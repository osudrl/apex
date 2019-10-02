import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentActor(nn.Module):
  def __init__(self, input_dim, action_dim, hidden_size=6, hidden_layers=1, has_critic=False):
    super(RecurrentActor, self).__init__()

    #self.actor_recurrent_layer = nn.LSTMCell(input_dim, hidden_size)

    self.actor_layers = nn.ModuleList()
    self.actor_layers += [nn.LSTMCell(input_dim, hidden_size)]
    for _ in range(hidden_layers-1):
        self.actor_layers += [nn.LSTMCell(hidden_size, hidden_size)]

    self.actor_out = nn.Linear(hidden_size, action_dim)

# ref: https://discuss.pytorch.org/t/correct-way-to-declare-hidden-and-cell-states-of-lstm/15745/3

    #self.hidden = [torch.zeros(1, hidden_size) for 
    #self.cells = [torch.zeros(1, hidden_size)

    print(self.actor_layers[0].size)
    exit(1)

  def get_hidden_state(self):
    return self.hidden, self.cells

  def init_hidden_state(self):
    pass
    
  
  def forward(self, x):
    #print(x.size(0), h.size(0))
    self.cells, self.hidden = self.actor_recurrent_layer(x, self.get_hidden_state())
    
    print("GOT ", self.cells, self.hidden)
