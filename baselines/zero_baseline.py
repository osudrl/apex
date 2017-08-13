"""A 'baseline' that returns all zeros. Same as having no baseline."""
import torch


class ZeroBaseline():
    def predict(self, observations):
        return torch.zeros(observations.size()[0], 1)

    def fit(self, paths):
        pass
