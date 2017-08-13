from rllab.envs.box2d.cartpole_env import CartpoleEnv
from policies.gaussian_mlp import GaussianMLP
from algos.vpg import VPG
from utils.evaluation import renderpolicy

if __name__ == "__main__":
    env = CartpoleEnv()

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = GaussianMLP(obs_dim, action_dim, (64,))

    algo = VPG(env, policy)

    algo.train(10, 100, 100)

    input("press enter")

    renderpolicy(env, policy, 300)
