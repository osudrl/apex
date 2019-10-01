import torch.nn as nn

class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()
        self.env = None # Gym environment name string

    def forward(self, x):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        value, x = self(inputs)
        action = self.dist.sample(x, deterministic=deterministic)
        return value, action.detach()

    def evaluate(self, inputs):
        value, x = self(inputs)
        return value, self.dist.evaluate(x)

class RecurrentPolicy(nn.module):
    def __init__(self):
        super(Recurrent, self).__init__()
        self.env = None # Gym environment name string

"""
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.env = None # Gym environment name string

    def forward(self, x):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        value, x = self(inputs)
        action = self.dist.sample(x, deterministic=deterministic)
        return value, action.detach()

    def evaluate(self, inputs):
        value, x = self(inputs)
        return value, self.dist.evaluate(x)

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.env = None # Gym environment name string

    def forward(self, x):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        value, x = self(inputs)
        action = self.dist.sample(x, deterministic=deterministic)
        return value, action.detach()

    def evaluate(self, inputs):
        value, x = self(inputs)
        return value, self.dist.evaluate(x)
"""
