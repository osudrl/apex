import torch.nn as nn


class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        value, x = self(inputs)
        action = self.dist.sample(x, deterministic=deterministic)
        return value, action.detach()

    def evaluate(self, inputs):
        value, x = self(inputs)
        return value, self.dist.evaluate(x)
