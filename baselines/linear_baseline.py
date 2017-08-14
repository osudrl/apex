"""A variance reduction baseline for policy gradient algorithms."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class FeatureEncodingBaseline():
    """
    Linear time-varying feature encoding baseline from Duan et al.

    See: https://arxiv.org/pdf/1604.06778.pdf
    """

    def __init__(self, obs_dim, weight_decay=1e-5):
        self.weights = torch.zeros(obs_dim * 2 + 4, 1)
        self.weight_decay = weight_decay

    def encode(self, observations):
        obs = torch.clamp(observations, -10, 10).data
        t = torch.range(0, len(observations) - 1) / 100
        ones = torch.ones(t.size())

        t = t.unsqueeze(1)
        ones = ones.unsqueeze(1)

        features = torch.cat([obs, obs**2, t, t**2, t**3, ones], 1)
        return features

    def predict(self, observations):
        features = self.encode(observations)
        return features @ self.weights

    def fit(self, paths):
        A = torch.cat([self.encode(path["observations"]) for path in paths])
        B = torch.cat([path["returns"] for path in paths]).data
        eye = torch.eye(A.size()[1])

        # solves (ridge) regularized orginary least squares to fit weights
        self.weights, _ = torch.gels(A.t() @ B,
                                     A.t() @ A + self.weight_decay * eye)
