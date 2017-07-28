from envs.slip_env import SlipEnv
from policies.gaussian_mlp import GaussianMLP
from algos.dagger import DAgger

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dagger_itr", type=int, default=10,
                    help="number of iterations of DAgger")
parser.add_argument("--epochs", type=int, default=100,
                    help="number of optimization epochs")
parser.add_argument("--trj_len", type=int, default=10000,
                    help="maximum trajectory length")
args = parser.parse_args()

if __name__ == "__main__":
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
