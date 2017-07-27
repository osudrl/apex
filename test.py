import torch
import numpy as np

from slip_env import SlipEnv
from gaussian_mlp import GaussianMLP
from dagger import DAgger

import logging

if __name__ == "__main__":
    env = SlipEnv(0.001)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    learner = GaussianMLP(obs_dim, action_dim, (64,))
    algo = DAgger(env, learner, None)

    algo.train(10, 100, 10000)
