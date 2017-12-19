"""Python file for automatically running experiments from command line."""
import argparse

from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines import bench


from rl.utils import run_experiment
from rl.policies import GaussianMLP
from rl.algos import PPO

import gym
import torch
import os


#TODO: remove reliance on: Monitor, DummyVecEnv, VecNormalized
def make_env(env_id, seed, rank, log_dir):
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = bench.Monitor(env,
                            os.path.join(log_dir,
                            os.path.join(log_dir, 
                            str(rank))), 
                            allow_early_resets=True
        )
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


    renv = make_env("Walker2d-v1", args.seed, 0, "/tmp/gym/rl/")
    renv = DummyVecEnv([renv])
    renv = VecNormalize(renv, ret=False)

    #env.seed(args.seed)
    #torch.manual_seed(args.seed)

    obs_dim = env_fn().observation_space.shape[0]
    action_dim = env_fn().action_space.shape[0]

    policy = GaussianMLP(obs_dim, action_dim)

    algo = PPO(args=args)

    run_experiment(
        algo=algo,
        policy=policy,
        env_fn=env_fn,
        renv=renv,
        args=args,
        log=True,
        monitor=False,
        render=True
    )
