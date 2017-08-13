"""Python file for automatically running experiments from command line."""
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from policies.gaussian_mlp import GaussianMLP
from baselines.linear_baseline import FeatureEncodingBaseline
from algos.vpg import VPG
from utils.evaluation import renderpolicy
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--n_itr", type=int, default=100,
                    help="number of iterations of the learning algorithm")
parser.add_argument("--max_trj_len", type=int, default=100,
                    help="maximum trajectory length")
parser.add_argument("--n_trj", type=int, default=100,
                    help="number of sample trajectories per iteration")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed for experiment")
args = parser.parse_args()

if __name__ == "__main__":
    env = CartpoleEnv()
    #env.seed(args.seed)

    torch.manual_seed(args.seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = GaussianMLP(obs_dim, action_dim, (64,))
    baseline = FeatureEncodingBaseline(obs_dim)

    algo = VPG(env, policy, baseline=baseline)

    algo.train(
        n_itr=args.n_itr,
        n_trj=args.n_trj,
        max_trj_len=args.max_trj_len
    )

    input("press enter to view policy simulation")

    renderpolicy(env, policy, 300)
