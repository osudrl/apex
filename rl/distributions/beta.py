import math

import torch
import torch.nn as nn


class Beta(nn.Module):
    def __init__(self, action_dim):
        super(Beta, self).__init__()

        self.action_dim = action_dim

    def forward(self, alpha_beta):
        alpha = 1 + nn.SoftPlus()(alpha_beta[:, :self.action_dim])
        beta = 1 + nn.SoftPlus()(alpha_beta[:, self.action_dim:])
        return alpha, beta

    def sample(self, x, deterministic):
        if deterministic is False:
            action = self.evaluate(x).sample()
        else:
            # NOTE: is mean expected value for beta? should check this
            # E = alpha / (alpha + beta)
            return self.evaluate(x).mean

        return action

    def evaluate(self, x):
        alpha, beta = self(x)
        return torch.distributions.Beta(alpha, beta)
