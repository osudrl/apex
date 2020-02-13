import sys
sys.path.append("..") # Adds higher directory to python modules path.

from rl.utils import renderpolicy, rendermultipolicy, renderpolicy_speedinput, rendermultipolicy_speedinput
from cassie import CassieEnv

import torch

import numpy as np
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=str, default="./trained_models/ppo/Cassie-v0/7b7e24-seed0/")
args = parser.parse_args()

cassie_env = CassieEnv(clock_based=True, state_est=True)
policy = torch.load(args.policy + "actor.pt")
policy.eval()
renderpolicy_speedinput(cassie_env, policy, deterministic=True, dt=0.05, speedup = 2)