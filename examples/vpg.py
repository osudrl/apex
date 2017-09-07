"""Python file for automatically running experiments from command line."""
import argparse

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize

from rl.utils import run_experiment
from rl.policies import GaussianA2C
from rl.baselines import FeatureEncodingBaseline
from rl.algos import VPG, CPI

import gym
import torch


parser = argparse.ArgumentParser()

CPI.add_arguments(parser)

parser.add_argument("--seed", type=int, default=1,
                    help="RNG seed")
parser.add_argument("--logdir", type=str, default="/tmp/rl/experiments/",
                    help="Where to log diagnostics to")

args = parser.parse_args()

if __name__ == "__main__":
    #env = normalize(Walker2DEnv())

    env = gym.make("Walker2d-v1")

    #env.seed(args.seed)
    #torch.manual_seed(args.seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = GaussianA2C(obs_dim, action_dim, (32,))

    algo = CPI(
        env=env,
        policy=policy,
        lr=args.lr,
        tau=args.tau
    )

    run_experiment(
        algo=algo,
        args=args,
        log=True,
        monitor=False,
        render=True
    )
