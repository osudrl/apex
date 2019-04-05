"""Python file for automatically running experiments from command line."""
import argparse

#from baselines import bench

from rl.utils import run_experiment
from rl.policies import GaussianMLP, BetaMLP
from rl.algos import PPO

from rl.envs.normalize import get_normalization_params, PreNormalizer

# NOTE: importing cassie for some reason breaks openai gym, BUG ?
from cassie import CassieEnv

#import gym
import torch

import numpy as np
import os

#TODO: remove reliance on: Monitor, DummyVecEnv, VecNormalized
# def make_env(env_id, seed, rank, log_dir):
#     def _thunk(log=True):
#         env = gym.make(env_id)
#         env.seed(seed + rank)
#         filename = os.path.join(log_dir,os.path.join(log_dir,str(rank))) \
#                    if log else None

#         #env = bench.Monitor(env, filename, allow_early_resets=True)
#         return env

#     return _thunk
from functools import partial

def make_cassie_env(*args, **kwargs):
    def _thunk():
        return CassieEnv(*args, **kwargs)
    return _thunk

parser = argparse.ArgumentParser()

PPO.add_arguments(parser)

parser.add_argument("--seed", type=int, default=1,
                    help="RNG seed")
parser.add_argument("--logdir", type=str, default="/tmp/rl/experiments/",
                    help="Where to log diagnostics to")
parser.add_argument("--name", type=str, default="model")

args = parser.parse_args()

#args.batch_size = 128 # Xie
args.batch_size = 256 # Peng
args.lr = 1e-4 # Xie
#args.lr = 5e-5 # Peng
args.epochs = 3 # Xie
#args.epochs = 5
#args.num_steps = 3000

args.num_procs = 30
args.num_steps = 500 #// args.num_procs
#args.num_steps = 3000 # Peng

args.max_traj_len = 400

args.use_gae = False

args.name = "Ray2"

# TODO: add ability to select graphs by number
# Interactive graphs/switch to tensorboard?
# More detailed logging
# Logging timestamps

if __name__ == "__main__":
    torch.set_num_threads(1) # see: https://github.com/pytorch/pytorch/issues/13757 

    #env_fn = make_env("Walker2d-v1", args.seed, 1337, "/tmp/gym/rl/")
    #env_fn = make_cassie_env("walking", clock_based=True)
    env_fn = partial(CassieEnv, "walking", clock_based=True)

    #env_fn = CassieTSEnv

    #env.seed(args.seed)
    #torch.manual_seed(args.seed)

    obs_dim = env_fn().observation_space.shape[0] # TODO: could make obs and ac space static properties
    action_dim = env_fn().action_space.shape[0]

    policy = GaussianMLP(obs_dim, action_dim, nonlinearity="tanh", init_std=np.exp(-2), learn_std=False)

    policy.obs_mean, policy.obs_std = map(torch.Tensor, get_normalization_params(iter=10000, noise_std=1, policy=policy, env_fn=env_fn))
    policy.train(0)

    #policy = BetaMLP(obs_dim, action_dim, nonlinearity="relu", init_std=np.exp(-2), learn_std=False)
    #normalizer = PreNormalizer(iter=10000, noise_std=1, policy=policy, online=False)

    algo = PPO(args=vars(args))

    # TODO: make log, monitor and render command line arguments
    # TODO: make algos take in a dictionary or list of quantities to log (e.g. reward, entropy, kl div etc)
    run_experiment(
        algo=algo,
        policy=policy,
        env_fn=env_fn,
        args=args,
        log=True,
        monitor=True,
        render=False # NOTE: CassieVis() hangs when launched in seperate thread. BUG?
                    # Also, waitpid() hangs on patrick's desktop in mp.Process. BUG?
    )
