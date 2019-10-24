"""Python file for automatically running experiments from command line."""
import argparse

from apex import print_logo

from rl.envs.wrappers import SymmetricEnv
from rl.utils import run_experiment
from rl.policies import GaussianMLP, BetaMLP
from rl.algos import PPO, MirrorPPO

from rl.envs.normalize import get_normalization_params, PreNormalizer

import functools

import torch

import numpy as np
import os

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from colorama import Fore, Style

def make_env_fn(state_est=False):
    def _thunk():
        return CassieEnv("walking", clock_based=True, state_est=state_est)
    return _thunk


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


# Arguments
parser = argparse.ArgumentParser()
#PPO.add_arguments(parser)
parser.add_argument("--redis_address", type=str, default=None)                  # address of redis server (for cluster setups)
parser.add_argument("--seed", type=int, default=1,help="RNG seed")

# For tensorboard logger
parser.add_argument("--logdir", type=str, default="./logs/ppo/experiments/")       # Where to log diagnostics to

parser.add_argument("--name", type=str, default="model")
parser.add_argument("--env", type=str, default="Cassie-mimic-walking-v0")
parser.add_argument("--state_est", type=bool, default=True)
# mirror actions or not
parser.add_argument("--mirror", default=False, action='store_true')
# visdom server port
parser.add_argument("--viz_port", default=8097)
# PPO algo args
parser.add_argument("--input_norm_steps", type=int, default=10000)
parser.add_argument("--n_itr", type=int, default=10000, help="Number of iterations of the learning algorithm")
parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate") # Xie
parser.add_argument("--eps", type=float, default=1e-5, help="Adam epsilon (for numerical stability)")
parser.add_argument("--lam", type=float, default=0.95, help="Generalized advantage estimate discount")
parser.add_argument("--gamma", type=float, default=0.99, help="MDP discount")
parser.add_argument("--entropy_coeff", type=float, default=0.0, help="Coefficient for entropy regularization")
parser.add_argument("--clip", type=float, default=0.2, help="Clipping parameter for PPO surrogate loss")
parser.add_argument("--minibatch_size", type=int, default=64, help="Batch size for PPO updates")
parser.add_argument("--epochs", type=int, default=3, help="Number of optimization epochs per PPO update") #Xie
parser.add_argument("--num_steps", type=int, default=5096, help="Number of sampled timesteps per gradient estimate")
parser.add_argument("--use_gae", type=bool, default=True,help="Whether or not to calculate returns using Generalized Advantage Estimation")
parser.add_argument("--num_procs", type=int, default=30, help="Number of threads to train on")
parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Value to clip gradients at.")
parser.add_argument("--max_traj_len", type=int, default=400, help="Max episode horizon")
args = parser.parse_args()
args.num_steps = 3000 // args.num_procs

if __name__ == "__main__":
    torch.set_num_threads(1) # see: https://github.com/pytorch/pytorch/issues/13757

    print_logo(subtitle="Distributed Proximal Policy Optimization")

    experiment_name = "PPO_{}_{}".format(args.env, args.num_procs)

    # Tensorboard logging
    now = datetime.now()
    # NOTE: separate by trial name first and time of run after
    log_path = args.logdir + now.strftime("%Y%m%d-%H%M%S")+"/"
    logger = SummaryWriter(log_path, flush_secs=0.1)
    print(Fore.GREEN + Style.BRIGHT + "Logging data using TensorBoard to {}".format(log_path + Style.RESET_ALL))

    # Environment
    if(args.env in ["Cassie-v0", "Cassie-mimic-v0", "Cassie-mimic-walking-v0"]):
        # NOTE: importing cassie for some reason breaks openai gym, BUG ?
        from cassie import CassieEnv, CassieTSEnv, CassieIKEnv
        from cassie.no_delta_env import CassieEnv_nodelta
        from cassie.speed_env import CassieEnv_speed
        from cassie.speed_double_freq_env import CassieEnv_speed_dfreq
        from cassie.speed_no_delta_env import CassieEnv_speed_no_delta
        # set up cassie environment
        # import gym_cassie
        # env_fn = gym_factory(args.env_name)
        #env_fn = make_env_fn(state_est=args.state_est)
        #env_fn = functools.partial(CassieEnv_speed_dfreq, "walking", clock_based = True, state_est=args.state_est)
        env_fn = functools.partial(CassieIKEnv, clock_based=True, state_est=args.state_est)
        print(env_fn().clock_inds)
        obs_dim = env_fn().observation_space.shape[0]
        action_dim = env_fn().action_space.shape[0]

        # Mirror Loss
        if args.mirror:
            if args.state_est:
                # with state estimator
                env_fn = functools.partial(SymmetricEnv, env_fn, mirrored_obs=[0, 1, 2, 3, 4, -10, -11, 12, 13, 14, -5, -6, 7, 8, 9, 15, 16, 17, 18, 19, 20, -26, -27, 28, 29, 30, -21, -22, 23, 24, 25, 31, 32, 33, 37, 38, 39, 34, 35, 36, 43, 44, 45, 40, 41, 42, 46, 47, 48], mirrored_act=[0,1,2,3,4,5,6,7,8,9])
            else:
                # without state estimator
                env_fn = functools.partial(SymmetricEnv, env_fn, mirrored_obs=[0, 1, 2, 3, 4, 5, -13, -14, 15, 16, 17,
                                                18, 19, -6, -7, 8, 9, 10, 11, 12, 20, 21, 22, 23, 24, 25, -33,
                                                -34, 35, 36, 37, 38, 39, -26, -27, 28, 29, 30, 31, 32, 40, 41, 42],
                                                mirrored_act = [0,1,2,3,4,5,6,7,8,9])
    else:
        import gym
        env_fn = gym_factory(args.env_name)
        #max_episode_steps = env_fn()._max_episode_steps
        obs_dim = env_fn().observation_space.shape[0]
        action_dim = env_fn().action_space.shape[0]
        max_episode_steps = 1000

    #env.seed(args.seed)
    #torch.manual_seed(args.seed)

    policy = GaussianMLP(
        obs_dim, action_dim, 
        nonlinearity="relu", 
        bounded=True, 
        init_std=np.exp(-2), 
        learn_std=False,
        normc_init=False
    )

    policy.obs_mean, policy.obs_std = map(torch.Tensor, get_normalization_params(iter=args.input_norm_steps, noise_std=1, policy=policy, env_fn=env_fn))

    # policy = torch.load("./trained_models/rom_tracking_v3.pt")
    # print("loaded model from {}".format("./trained_models/rom_tracking_v3.pt"))

    policy.train(0)

    print("obs_dim: {}, action_dim: {}".format(obs_dim, action_dim))

    if args.mirror:
        algo = MirrorPPO(args=vars(args))
    else:
        algo = PPO(args=vars(args))
        
    # TODO: make log, monitor and render command line arguments
    # TODO: make algos take in a dictionary or list of quantities to log (e.g. reward, entropy, kl div etc)
    run_experiment(
        algo=algo,
        policy=policy,
        env_fn=env_fn,
        experiment_name=experiment_name,
        args=args,
        logger=logger,
        monitor=True,
        render=False # NOTE: CassieVis() hangs when launched in seperate thread. BUG?
                    # Also, waitpid() hangs sometimes in mp.Process. BUG?
    )
