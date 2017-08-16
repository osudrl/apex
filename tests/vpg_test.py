"""Python file for automatically running experiments from command line."""
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize

from utils.experiment import run_experiment
from policies.gaussian_mlp import GaussianMLP
from baselines.linear_baseline import FeatureEncodingBaseline
from algos.vpg import VPG

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--n_itr", type=int, default=1000,
                    help="number of iterations of the learning algorithm")
parser.add_argument("--max_trj_len", type=int, default=200,
                    help="maximum trajectory length")
parser.add_argument("--n_trj", type=int, default=100,
                    help="number of sample trajectories per iteration")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed for experiment")
parser.add_argument("--lr", type=int, default=0.01,
                    help="Adam learning rate")
parser.add_argument("--desired_kl", type=int, default=0.01,
                    help="Desired change in mean kl per iteration")
args = parser.parse_args()

if __name__ == "__main__":
    #env = normalize(DoublePendulumEnv())
    #env = normalize(CartpoleEnv())
    env = normalize(Walker2DEnv())
    #env.seed(args.seed)

    #torch.manual_seed(args.seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = GaussianMLP(obs_dim, action_dim, (8,))
    baseline = FeatureEncodingBaseline(obs_dim)

    algo = VPG(env=env, policy=policy, baseline=baseline, lr=args.lr)

    run_experiment(algo, args, render=True)
