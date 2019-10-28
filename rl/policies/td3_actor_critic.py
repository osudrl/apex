import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)



# By default all the modules are initialized to train mode (self.training = True)

class Original_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_size1, hidden_size2, init_w=3e-3):
        super(Original_Actor, self).__init__()

        self.max_action = max_action

        self.l1 = nn.Linear(state_dim, hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, action_dim)
    
    def init_weights(self, init_w):
        self.l1.weight.data = fanin_init(self.l1.weight.data.size())
        self.l2.weight.data = fanin_init(self.l2.weight.data.size())
        self.l3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

class DDPGCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size1, hidden_size2, init_w=3e-3):
        super(DDPGCritic, self).__init__()

        self.max_action = max_action

        self.l1 = nn.Linear(state_dim + action_dim, hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, 1)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.l1.weight.data = fanin_init(self.l1.weight.data.size())
        self.l2.weight.data = fanin_init(self.l2.weight.data.size())
        self.l3.weight.data.uniform_(-init_w, init_w)

    def forward(self, inputs, actions):
        xu = torch.cat([inputs, actions], 1)
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        return x1

class TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size1, hidden_size2, init_w=3e-3):
        super(TD3Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_size1)
        self.l5 = nn.Linear(hidden_size1, hidden_size2)
        self.l6 = nn.Linear(hidden_size2, 1)

        # init weights for both nets
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.l1.weight.data = fanin_init(self.l1.weight.data.size())
        self.l2.weight.data = fanin_init(self.l2.weight.data.size())
        self.l3.weight.data.uniform_(-init_w, init_w)

        self.l4.weight.data = fanin_init(self.l4.weight.data.size())
        self.l5.weight.data = fanin_init(self.l5.weight.data.size())
        self.l6.weight.data.uniform_(-init_w, init_w)

    def forward(self, inputs, actions):
        xu = torch.cat([inputs, actions], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)

        return x1, x2

    def Q1(self, inputs, actions):
        xu = torch.cat([inputs, actions], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        return x1

# Layernorm (marked by LN) used to make correlated parameter noise possible for DDPG and TD3
class LN_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_size1, hidden_size2, init_w=3e-3):
        super(LN_Actor, self).__init__()

        self.max_action = max_action

        self.l1 = nn.Linear(state_dim, hidden_size1)
        self.ln1 = nn.LayerNorm(hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.ln2 = nn.LayerNorm(hidden_size2)
        self.l3 = nn.Linear(hidden_size2, action_dim)

        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.l1.weight.data = fanin_init(self.l1.weight.data.size())
        self.l2.weight.data = fanin_init(self.l2.weight.data.size())
        self.l3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.ln1(x)
        x = F.relu(self.l2(x))
        x = self.ln2(x)
        x = torch.tanh(self.l3(x))
        #x = self.max_action * torch.tanh(self.l3(x))
        return x


class LN_DDPGCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size1, hidden_size2, init_w=3e-3):
        super(LN_DDPGCritic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_size1)
        self.ln1 = nn.LayerNorm(hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.ln2 = nn.LayerNorm(hidden_size2)
        self.l3 = nn.Linear(hidden_size2, 1)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.l1.weight.data = fanin_init(self.l1.weight.data.size())
        self.l2.weight.data = fanin_init(self.l2.weight.data.size())
        self.l3.weight.data.uniform_(-init_w, init_w)

    def forward(self, inputs, actions):
        xu = torch.cat([inputs, actions], 1)

        x1 = F.relu(self.l1(xu))
        x1 = self.ln1(x1)
        x1 = F.relu(self.l2(x1))
        x1 = self.ln2(x1)
        x1 = self.l3(x1)

        return x1

# critic uses 2 action-value functions (and uses smaller one to form targets)
class LN_TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size1, hidden_size2, init_w=3e-3):
        super(LN_TD3Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_size1)
        self.ln1 = nn.LayerNorm(hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.ln2 = nn.LayerNorm(hidden_size2)
        self.l3 = nn.Linear(hidden_size2, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_size1)
        self.ln4 = nn.LayerNorm(hidden_size1)
        self.l5 = nn.Linear(hidden_size1, hidden_size2)
        self.ln5 = nn.LayerNorm(hidden_size2)
        self.l6 = nn.Linear(hidden_size2, 1)

        # init weights for both nets
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.l1.weight.data = fanin_init(self.l1.weight.data.size())
        self.l2.weight.data = fanin_init(self.l2.weight.data.size())
        self.l3.weight.data.uniform_(-init_w, init_w)

        self.l4.weight.data = fanin_init(self.l4.weight.data.size())
        self.l5.weight.data = fanin_init(self.l5.weight.data.size())
        self.l6.weight.data.uniform_(-init_w, init_w)

    def forward(self, inputs, actions):
        xu = torch.cat([inputs, actions], 1)

        x1 = F.relu(self.l1(xu))
        x1 = self.ln1(x1)
        x1 = F.relu(self.l2(x1))
        x1 = self.ln2(x1)
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = self.ln4(x2)
        x2 = F.relu(self.l5(x2))
        x2 = self.ln5(x2)
        x2 = self.l6(x2)

        return x1, x2

    def Q1(self, inputs, actions):
        xu = torch.cat([inputs, actions], 1)

        x1 = F.relu(self.l1(xu))
        x1 = self.ln1(x1)
        x1 = F.relu(self.l2(x1))
        x1 = self.ln2(x1)
        x1 = self.l3(x1)

        return x1
