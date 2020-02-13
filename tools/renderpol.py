import sys
sys.path.append("..") # Adds higher directory to python modules path.

from rl.utils import renderpolicy, rendermultipolicy, renderpolicy_speedinput, rendermultipolicy_speedinput
from cassie import CassieEnv

import torch

import numpy as np
import os
import time

import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="./trained_models/ppo/Cassie-v0/7b7e24-seed0/", help="path to folder containing policy and run details")
args = parser.parse_args()
run_args = pickle.load(open(args.path + "experiment.pkl", "rb"))

cassie_env = CassieEnv(traj=run_args.traj, clock_based=run_args.clock_based, state_est=run_args.state_est, dynamics_randomization=run_args.dyn_random)
policy = torch.load(args.path + "actor.pt")
policy.eval()

renderpolicy_speedinput(cassie_env, policy, deterministic=True, dt=0.05, speedup = 2)