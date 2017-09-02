"""Python file for automatically running experiments from command line."""
import argparse

#from rllab.envs.box2d.cartpole_env import CartpoleEnv
#from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
#from rllab.envs.mujoco.walker2d_env import Walker2DEnv
#from rllab.envs.mujoco.hopper_env import HopperEnv
#from rllab.envs.mujoco.walker2d_env import Walker2DEnv
#from rllab.envs.gym_env import GymEnv
#from rllab.envs.normalized_env import normalize

from rl.utils import run_experiment
from rl.policies import GaussianA2C
from rl.algos import PPO

import gym
import torch


parser = argparse.ArgumentParser()

PPO.add_arguments(parser)

parser.add_argument("--seed", type=int, default=1,
                    help="RNG seed")
parser.add_argument("--logdir", type=str, default="/tmp/rl/experiments/",
                    help="Where to log diagnostics to")

args = parser.parse_args()

"""
This is how schulman et al initialize their weights.

def normc_init(m):
    if isinstance(m, torch.nn.Linear):
        out = torch.randn(m.weight.data.size())
        out *= 1 / (out.pow(2)).sum(dim=1, keepdim=True).sqrt()
        m.weight.data = out
        m.bias.data *= 0
"""

if __name__ == "__main__":
    #env = normalize(Walker2DEnv())
    #env = normalize(GymEnv("Walker2d-v1"))

    env = gym.make("Walker2d-v1")

    #env.seed(args.seed)
    #torch.manual_seed(args.seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = GaussianA2C(obs_dim, action_dim, (32,))

    """ Only works in pytorch 0.2
    policy.apply(normc_init)
    """

    algo = PPO(
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
        render=False
    )
