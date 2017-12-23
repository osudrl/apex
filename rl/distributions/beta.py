import math

import torch
import torch.nn as nn
from torch.autograd import Variable


class Beta(nn.Module):
    def __init__(self):
        super(Beta, self).__init__()

    def forward(self, alpha, beta):
        return alpha, beta

    def sample(self, x, deterministic):
        if deterministic is False:
            action = self.evaluate(x).sample()
        else:
            alpha, beta = self(x)
            return alpha / (alpha + beta)

        return action

    def evaluate(self, x):
        alpha, beta = self(x)
        return torch.distributions.Beta(alpha, beta)
