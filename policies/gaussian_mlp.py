import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# MLP policy that outputs a mean and samples actions from a gaussian
# distribution with a learned (but fixed in space) std dev
class GaussianMLP(nn.Module):
    def __init__(self, obs_dim, action_dim,  hidden_dims=(128, 128),
                 init_std=1.0, nonlin=F.tanh, optimizer=optim.Adam):

        super(GaussianMLP, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers += [nn.Linear(obs_dim, hidden_dims[0])]
        for dim in hidden_dims:
            self.hidden_layers += [nn.Linear(dim, dim)]

        self.out = nn.Linear(hidden_dims[-1], action_dim)

        self.log_stds = nn.Parameter(
            torch.ones(1, action_dim) * init_std
        )

        self.nonlin = nonlin

        print(self.nonlin)

        self.optimizer = optimizer(self.parameters(),
                                   lr=0.003,
                                   weight_decay=0.0)

        print(self)

    def forward(self, x):
        for l in self.hidden_layers:
            x = self.nonlin(l(x))

        means = self.out(x)

        log_stds = self.log_stds.repeat(x.size()[0], 1)  # batch size compensation

        rnd = Variable(torch.randn(log_stds.size()))

        actions = rnd * torch.exp(log_stds) + means

        return actions

    def loss(self, X, y):
        output = self.forward(X)
        loss = nn.MSELoss()(output, y)
        return loss

    def fit(self, X, y):
        self.optimizer.zero_grad()
        output = self.forward(X)
        loss = nn.MSELoss()(output, y)
        loss.backward()  # accumulate gradients
        self.optimizer.step()  # update parameters

        return loss.data
