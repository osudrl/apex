"""Python file for automatically running experiments from command line."""
import argparse

#from baselines import bench

from rl.utils import run_experiment
from rl.policies import GaussianMLP, BetaMLP
from rl.algos import PPO

import gym
import torch
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

parser = argparse.ArgumentParser()

PPO.add_arguments(parser)

parser.add_argument("--seed", type=int, default=1,
                    help="RNG seed")
parser.add_argument("--logdir", type=str, default="/tmp/rl/experiments/",
                    help="Where to log diagnostics to")

args = parser.parse_args()

if __name__ == "__main__":
    env_fn = make_env("Walker2d-v1", args.seed, 1337, "/tmp/gym/rl/")

    #env.seed(args.seed)
    #torch.manual_seed(args.seed)

    obs_dim = env_fn().observation_space.shape[0]
    action_dim = env_fn().action_space.shape[0]

    policy = GaussianMLP(obs_dim, action_dim)
    #policy = BetaMLP(obs_dim, action_dim)

    algo = PPO(args=args)

    run_experiment(
        algo=algo,
        policy=policy,
        env_fn=env_fn,
        args=args,
        log=False,
        monitor=False,
        render=True
    )
