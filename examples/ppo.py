"""Python file for automatically running experiments from command line."""
import argparse

#from baselines import bench

from rl.utils import run_experiment
from rl.policies import GaussianMLP, BetaMLP
from rl.algos import PPO

from rl.envs.normalize import PreNormalizer

# NOTE: importing cassie for some reason breaks openai gym, BUG ?
from cassie import CassieEnv

#import gym
import torch

import numpy as np
import os

#TODO: remove reliance on: Monitor, DummyVecEnv, VecNormalized
def make_env(env_id, seed, rank, log_dir):
    def _thunk(log=True):
        env = gym.make(env_id)
        env.seed(seed + rank)
        filename = os.path.join(log_dir,os.path.join(log_dir,str(rank))) \
                   if log else None

        #env = bench.Monitor(env, filename, allow_early_resets=True)
        return env

    return _thunk

def make_cassie_env(traj_dir):
    def _thunk():
        return CassieEnv(traj_dir)
    return _thunk

parser = argparse.ArgumentParser()

PPO.add_arguments(parser)

parser.add_argument("--seed", type=int, default=1,
                    help="RNG seed")
parser.add_argument("--logdir", type=str, default="/tmp/rl/experiments/",
                    help="Where to log diagnostics to")
parser.add_argument("--name", type=str, default="model")

args = parser.parse_args()

# Xie et al params:
# batch_size = 128, hardcoded to 128
# lr = 1e-3, but gets set to 1e-4 in the code
# num_epoch = 32, hardcoded to 64
# his "epochs" are actually minibatch optimization steps...
# which look to be done with replacement
# equivalent epochs is 64*128/3000 ~= 3

# num_steps = 2048, but gets multipled by 10,000?
# ^ but samples are hardcoded to 300 in the code...
# ^ but memory that optimization gets drawn from is set to 
# ^ 3000...
# max_episode_length = 2048 # not used
# min_episode = 5           # not used
# time_horizon = 200,000    # not used

# differences between Xie et al RL and my RL:
# Xie et al: no GAE, different normalization scheme, seperate optimization for actor and critic, learning rate scheduling
# Xie et al does not center advantages or rewards
# optimizer reset every run
# learning rate scheduling not used

# policy size: (256, 256)

# true hyperparams: 
args.batch_size = 128
args.lr = 1e-4
args.epochs = 3
args.num_steps = 3000

args.use_gae = False

args.name = "Xie3"

if __name__ == "__main__":
    torch.set_num_threads(1) # see: https://github.com/pytorch/pytorch/issues/13757 

    #env_fn = make_env("Walker2d-v1", args.seed, 1337, "/tmp/gym/rl/")

    env_fn = make_cassie_env("walking")

    #env.seed(args.seed)
    #torch.manual_seed(args.seed)

    obs_dim = env_fn().observation_space.shape[0] # TODO: could make obs and ac space static properties
    action_dim = env_fn().action_space.shape[0]

    policy = GaussianMLP(obs_dim, action_dim, nonlinearity="relu", init_std=np.exp(-2), learn_std=False)
    #policy = BetaMLP(obs_dim, action_dim)

    normalizer = None #PreNormalizer(iter=10000, noise_std=1, policy=policy, online=False)

    algo = PPO(args=vars(args))

    # TODO: make log, monitor and render command line arguments
    # TODO: make algos take in a dictionary or list of quantities to log (e.g. reward, entropy, kl div etc)
    run_experiment(
        algo=algo,
        policy=policy,
        env_fn=env_fn,
        normalizer=normalizer,
        args=args,
        log=True,
        monitor=True,
        render=False # NOTE: CassieVis() hangs when launched in seperate thread. BUG?
                     # Also, waitpid() hangs on patrick's desktop in mp.Process. BUG?
    )
