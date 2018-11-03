"""Python file for automatically running experiments from command line."""
import argparse

#from baselines import bench

from rl.utils import run_experiment
from rl.policies import GaussianMLP, BetaMLP
from rl.algos import PPO

#from cassieXie.simple_env import cassieRLEnv

# NOTE: importing cassie for some reason breaks openai gym, BUG ?
from cassie import CassieEnv

#import gym
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
parser.add_argument("--name", type=str, default="model1")

args = parser.parse_args()

if __name__ == "__main__":
    #env_fn = make_env("Walker2d-v1", args.seed, 1337, "/tmp/gym/rl/")

    env_fn = make_cassie_env("cassie/trajectory/stepdata.bin")

    #env.seed(args.seed)
    #torch.manual_seed(args.seed)

    obs_dim = env_fn().observation_space.shape[0] # TODO: could make obs and ac space static properties
    action_dim = env_fn().action_space.shape[0]

    policy = GaussianMLP(obs_dim, action_dim)
    #policy = BetaMLP(obs_dim, action_dim)

    algo = PPO(args=args)


    # Zhaoming params:
    # batch_size = 128
    # lr = 1e-3
    # num_epoch = 32
    # num_steps = 2048
    # max_episode_length = 2048 # not used
    # min_episode = 5           # not used
    # time_horizon = 200,000    # not used

    # differences between Zhaoming RL and my RL:
    # Zhaoming: no GAE, different normalization scheme, seperate optimization for actor and critic, learning rate scheduling


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
