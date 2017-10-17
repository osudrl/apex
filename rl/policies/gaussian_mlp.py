import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.distributions import DiagonalGaussian
from .running_stat import ObsNorm
from .base import FFPolicy

def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class GaussianMLP(FFPolicy):
    def __init__(self, num_inputs, action_space, init_std=1):
        super(GaussianMLP, self).__init__()

        actor_dims = (64,)
        critic_dims = (64, 64)

        self.obs_filter = ObsNorm((1, num_inputs), clip=5)

        self.action_space = action_space

        # create actor network
        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.Linear(num_inputs, actor_dims[0])]
        for l in range(len(actor_dims) - 1):
            in_dim = actor_dims[l]
            out_dim = actor_dims[l + 1]
            self.actor_layers += [nn.Linear(in_dim, out_dim)]

        # create critic network
        self.critic_layers = nn.ModuleList()
        self.critic_layers += [nn.Linear(num_inputs, critic_dims[0])]
        for l in range(len(critic_dims) - 1):
            in_dim = critic_dims[l]
            out_dim = critic_dims[l + 1]
            self.critic_layers += [nn.Linear(in_dim, out_dim)]

        self.vf = nn.Linear(critic_dims[-1], 1)


        if action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagonalGaussian(64, num_outputs)

        elif action_space.__class__.__name__ == "Discrete":
            #num_outputs = action_space.n
            #self.dist = Categorical(64, num_outputs)
            raise NotImplementedError

        else:
            raise NotImplementedError

        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init_mlp)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def cuda(self, **args):
        super(GaussianMLP, self).cuda(**args)
        self.obs_filter.cuda()

    def cpu(self, **args):
        super(GaussianMLP, self).cpu(**args)
        self.obs_filter.cpu()

    def forward(self, inputs):
        inputs.data = self.obs_filter(inputs.data)

        x = inputs
        for l in self.critic_layers:
            x = F.tanh(l(x))
        value = self.vf(x)

        x = inputs
        for l in self.actor_layers:
            x = F.tanh(l(x))

        return value, x
