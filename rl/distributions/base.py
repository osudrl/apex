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

    def likelihood_ratio(self, x, new):
        old_logli = self.log_likelihood(x)
        new_logli = new.log_likelihood(x)
        return torch.exp(new_logli - old_logli)
