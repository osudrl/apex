"""Python file for automatically running experiments from command line."""
import argparse

from rl.utils import run_experiment
from rl.policies import GaussianMLP, BetaMLP
from rl.algos import PPO

from rl.envs.normalize import get_normalization_params, PreNormalizer

import torch

import numpy as np
import os

parser = argparse.ArgumentParser()

PPO.add_arguments(parser)

parser.add_argument("--seed", type=int, default=1,
                    help="RNG seed")
parser.add_argument("--logdir", type=str, default="/tmp/rl/experiments/",
                    help="Where to log diagnostics to")
parser.add_argument("--name", type=str, default="model")

parser.add_argument("--env", type=str, default="Cassie-mimic-walking-v0")

# visdom server port
parser.add_argument("--viz_port", default=8097)                                 

parser.add_argument("--input_norm_steps", type=int, default=10000)

args = parser.parse_args()

#args.batch_size = 128 # Xie
args.batch_size = 256 # Peng
args.lr = 1e-4 # Xie
#args.lr = 5e-5 # Peng
args.epochs = 3 # Xie
#args.epochs = 5

args.num_procs = 30
args.num_steps = 3000 // args.num_procs

#args.num_steps = 500 #// args.num_procs
#args.num_steps = 3000 # Peng

args.max_traj_len = 400

args.use_gae = False

args.name = "demo2"

# TODO: add ability to select graphs by number
# Interactive graphs/switch to tensorboard?
# More detailed logging
# Logging timestamps

#import gym
#import gym_cassie
from cassie import CassieEnv

def gym_factory(path, **kwargs):
    from functools import partial

    """
    This is (mostly) equivalent to gym.make(), but it returns an *uninstantiated* 
    environment constructor.

    Since environments containing cpointers (e.g. Mujoco envs) can't be serialized, 
    this allows us to pass their constructors to Ray remote functions instead 
    (since the gym registry isn't shared across ray subprocesses we can't simply 
    pass gym.make() either)

    Note: env.unwrapped.spec is never set, if that matters for some reason.
    """
    spec = gym.envs.registry.spec(path)
    _kwargs = spec._kwargs.copy()
    _kwargs.update(kwargs)
    
    if callable(spec._entry_point):
        cls = spec._entry_point(**_kwargs)
    else:
        cls = gym.envs.registration.load(spec._entry_point)

    return partial(cls, **_kwargs)

def make_env_fn():
    def _thunk():
        return CassieEnv("walking", clock_based=True)
    return _thunk


if __name__ == "__main__":
    torch.set_num_threads(1) # see: https://github.com/pytorch/pytorch/issues/13757 

    #env_fn = gym_factory(args.env)  # for use with gym_cassie
    env_fn = make_env_fn()

    #env.seed(args.seed)
    #torch.manual_seed(args.seed)

    obs_dim = env_fn().observation_space.shape[0] 
    action_dim = env_fn().action_space.shape[0]

    policy = GaussianMLP(
        obs_dim, action_dim, 
        nonlinearity="relu", 
        bounded=True, 
        init_std=np.exp(-2), 
        learn_std=False,
        normc_init=False
    )

    policy.obs_mean, policy.obs_std = map(torch.Tensor, get_normalization_params(iter=args.input_norm_steps, noise_std=1, policy=policy, env_fn=env_fn))
    policy.train(0)

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
                    # Also, waitpid() hangs sometimes in mp.Process. BUG?
    )
