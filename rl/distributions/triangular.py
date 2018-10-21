import math

import torch
import torch.nn as nn
from torch.autograd import Variable


class Beta(nn.Module):
    def __init__(self, minimum, maximum):
        super(Beta, self).__init__()

        self.min = minimum
        self.max = maximum

    def forward(self, mode):
        return self.min, self.max, mode

    def sample(self, x, deterministic):
        if deterministic is False:
            action = self.evaluate(x).sample()
        else:
            return

        return action

    def evaluate(self, x):
        a, b, c = self(x)
        return Triangular(a, b, c)

class Triangular:
    def __init__(a, b, c):
        self.a = a
        self.b = b
        self.c = c
    
    def log_prob(x):
        if 