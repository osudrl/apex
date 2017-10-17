import math

import torch
import torch.nn as nn


class DiagonalGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, init_std=1):
        super(DiagonalGaussian, self).__init__()
        self.fc_mean = nn.Linear(64, num_outputs)

        self.logstd = nn.Parameter(
            torch.ones(1, num_outputs) * math.log(init_std)
        )

    def forward(self, x):
        x = self.fc_mean(x)
        action_mean = x

        x = self.logstd
        action_logstd = x

        return action_mean, action_logstd

    def sample(self, x, deterministic):
        action_mean, action_logstd = self(x)

        action_std = action_logstd.exp()

        if deterministic is False:
            action = torch.normal(action_mean, action_std)
        else:
            action = action_mean

        return action

    def logp(self, x, mean, std, log_std):
        logp = -0.5 * ((x - mean) / std).pow(2) - 0.5 * math.log(2 * math.pi) - log_std
        return logp.sum(1, keepdim=True)

    def entropy(self, logp):
        return (0.5 + math.log(2 * math.pi) + logp).sum(-1).mean()

    def evaluate_actions(self, x, actions):
        action_mean, action_logstd = self(x)

        action_std = action_logstd.exp()

        action_log_probs = self.logp(actions, action_mean, action_std, action_logstd)
        dist_entropy = self.entropy(action_log_probs)

        return action_log_probs, dist_entropy
