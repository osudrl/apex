import math

import torch
import torch.nn as nn
from torch.autograd import Variable


class Beta(nn.Module):
    def __init__(self, action_dim):
        super(Beta, self).__init__()

        self.action_dim = action_dim

    def forward(self, alpha_beta):
        alpha = alpha_beta[0, :self.action_dim]
        beta = alpha_beta[0, self.action_dim:]
        return alpha, beta

    def sample(self, x, deterministic):
        if deterministic is False:
            action = self.evaluate(x).sample()
        else:
            alpha, beta = self(x)
            # expected value of a beta distribution:
            return alpha / (alpha + beta)

        return action

    def evaluate(self, x):
        alpha, beta = self(x)
        return torch.distributions.Beta(alpha, beta)
