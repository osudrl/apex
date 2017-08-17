class Distribution():
    def entropy(self, distribution):
        raise NotImplementedError

    def kl_divergence(self, distribution0, distribution1):
        raise NotImplementedError

    def log_likelihood(self, x, distribution):
        raise NotImplementedError
