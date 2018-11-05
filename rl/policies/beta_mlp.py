import torch
import torch.nn as nn
import torch.functional as F

from rl.distributions import Beta
from .base import FFPolicy

def normc_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class BetaMLP(FFPolicy):
    def __init__(self, num_inputs, action_dim, nonlinearity="tanh"):
        super(BetaMLP, self).__init__()

        actor_dims = (64, 64)
        critic_dims = (64, 64)

        # create actor network
        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.Linear(num_inputs, actor_dims[0])]
        for l in range(len(actor_dims) - 1):
            in_dim = actor_dims[l]
            out_dim = actor_dims[l + 1]
            self.actor_layers += [nn.Linear(in_dim, out_dim)]
        
        self.alpha_beta = nn.Linear(actor_dims[-1], 2 * action_dim)

        # create critic network
        self.critic_layers = nn.ModuleList()
        self.critic_layers += [nn.Linear(num_inputs, critic_dims[0])]
        for l in range(len(critic_dims) - 1):
            in_dim = critic_dims[l]
            out_dim = critic_dims[l + 1]
            self.critic_layers += [nn.Linear(in_dim, out_dim)]

        self.vf = nn.Linear(critic_dims[-1], 1)

        self.dist = Beta(action_dim)

        if nonlinearity == "relu":
            self.nonlinearity = F.relu
        else:
            self.nonlinearity = torch.tanh

        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(normc_init)

        #if self.dist.__class__.__name__ == "DiagGaussian":
        #    self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs):
        x = inputs
        for l in self.critic_layers:
            x = torch.tanh(l(x))
        value = self.vf(x)

        x = inputs
        for l in self.actor_layers:
            x = torch.tanh(l(x))
        x = self.alpha_beta(x)

        return value, x