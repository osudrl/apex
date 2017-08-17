from rl.envs import SlipEnv
from rl.policies import GaussianMLP
from rl.algos import DAgger
from rl.utils import policyplot

import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dagger_itr", type=int, default=100,
                    help="number of iterations of DAgger")
parser.add_argument("--epochs", type=int, default=10,
                    help="number of optimization epochs")
parser.add_argument("--trj_len", type=int, default=10000,
                    help="maximum trajectory length")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed for experiment")
args = parser.parse_args()

if __name__ == "__main__":
    torch.manual_seed(args.seed)

    env = SlipEnv(0.001)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    learner = GaussianMLP(obs_dim, action_dim, (64,))
    algo = DAgger(env, learner, None)

    algo.train(
        dagger_itr=args.dagger_itr,
        epochs=args.epochs,
        trj_len=args.trj_len
    )

    policyplot(env, learner, args.trj_len)
