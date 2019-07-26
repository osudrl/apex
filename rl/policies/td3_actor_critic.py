import torch
import torch.nn as nn
import torch.nn.functional as F

# By default all the modules are initialized to train mode (self.training = True)


class LN_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_size1, hidden_size2):
        super(LN_Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_size1)
        self.ln1 = nn.LayerNorm(hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.ln2 = nn.LayerNorm(hidden_size2)
        self.l3 = nn.Linear(hidden_size2, action_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.ln1(x)
        x = F.relu(self.l2(x))
        x = self.ln2(x)
        x = torch.tanh(self.l3(x))
        #x = self.max_action * torch.tanh(self.l3(x))
        return x

class LN_DDPGCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size1, hidden_size2):
        super(LN_DDPGCritic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_size1)
        self.ln1 = nn.LayerNorm(hidden_size1)

        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.ln2 = nn.LayerNorm(hidden_size2)

        self.l3 = nn.Linear(hidden_size2, 1)

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
    def __init__(self, state_dim, action_dim, hidden_size1, hidden_size2):
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