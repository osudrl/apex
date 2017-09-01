import torch

class Distribution():
    def entropy(self):
        raise NotImplementedError

    def kl_divergence(self, new):
        raise NotImplementedError

    def log_likelihood(self, x):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def likelihood_ratio(self, actions, old, new):
        old_logli = self.log_likelihood(actions, old)
        new_logli = self.log_likelihood(actions, new)
        return torch.exp(new_logli - old_logli)
