import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.distributions import Beta, Beta2
from .base import FFPolicy

def normc_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

# TODO: add a variance-mean parameterization of the Beta,
# to allow for training with a fixed variance
# e.g.: 
# alpha = ((1 - mu) / (sigma^2) - 1 / mu) * mu^2
# beta = alpha * (1 / mu - 1)
# with mu in (0, 1) and sigma^2 in (0, 0,5^2)
class BetaMLP(FFPolicy):
    def __init__(self, 
                 num_inputs, 
                 action_dim, 
                 nonlinearity="tanh", 
                 normc_init=False,
                 init_std=None,
                 learn_std=False):
        super(BetaMLP, self).__init__()

        actor_dims = (64, 64)
        critic_dims = (64, 64)

        if init_std is not None:
            self.dist = Beta2(action_dim, init_std, learn_std)
            num_outputs = action_dim
        else:
            self.dist = Beta(action_dim)
            num_outputs = 2 * action_dim

        # create actor network
        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.Linear(num_inputs, actor_dims[0])]
        for l in range(len(actor_dims) - 1):
            in_dim = actor_dims[l]
            out_dim = actor_dims[l + 1]
            self.actor_layers += [nn.Linear(in_dim, out_dim)]
        
        self.params = nn.Linear(actor_dims[-1], num_outputs)

        # create critic network
        self.critic_layers = nn.ModuleList()
        self.critic_layers += [nn.Linear(num_inputs, critic_dims[0])]
        for l in range(len(critic_dims) - 1):
            in_dim = critic_dims[l]
            out_dim = critic_dims[l + 1]
            self.critic_layers += [nn.Linear(in_dim, out_dim)]

        self.vf = nn.Linear(critic_dims[-1], 1)

        if nonlinearity == "relu":
            self.nonlinearity = F.relu
        else:
            self.nonlinearity = torch.tanh

        # weight initialization scheme used in PPO paper experiments
        self.normc_init = normc_init

        self.init_parameters()
        self.train()

    def init_parameters(self):
        if self.normc_init:
            self.apply(normc_init)

            if self.dist.__class__.__name__ == "DiagGaussian":
                self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs):
        x = inputs
        for l in self.critic_layers:
            x = torch.tanh(l(x))
        value = self.vf(x)

        x = inputs
        for l in self.actor_layers:
            x = torch.tanh(l(x))
        x = self.params(x)

        return value, x