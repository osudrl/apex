# TODO: organize this file
from rl.utils import select_action
from rl.envs.wrappers import SymmetricEnv
from rl.policies.td3_actor_critic import LN_Actor as Actor
import numpy as np
import argparse

import time
import os
import gym

import ray
import torch

import functools

import matplotlib
from matplotlib import pyplot as plt
plt.style.use('ggplot')

np.set_printoptions(precision=2, suppress=True)

device = torch.device('cpu')


def make_cassie_env(*args, **kwargs):
    def _thunk():
        return CassieEnv(*args, **kwargs)
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

# TODO: add .dt to all environments. OpenAI should do the same...


def visualize(env_fn, policy, vlen, dt=0.033, speedup=1):

    env = env_fn()

    done = False
    R = []
    episode_reward = 0
    state = torch.Tensor([env.reset()])
    t = 0

    with torch.no_grad():

        while True:
            t += 1
            start = time.time()
            action = select_action(policy, np.array(state), device)

            start = time.time()
            next_state, reward, done, _ = env.step(action)

            done_bool = 1.0 if t + 1 == vlen else float(done)

            episode_reward += reward

            next_state = torch.Tensor([next_state])

            state = next_state
            if done_bool:
                print(episode_reward)
                state = env.reset()
                R += [episode_reward]
                episode_reward = 0
                t = 0

            state = torch.Tensor(state)

            env.render()

            time.sleep(dt / speedup)

        if not done:
            R += [episode_reward]

        print("avg reward:", sum(R)/len(R))
        print("avg timesteps:", vlen / len(R))


ray.init(num_gpus=0)

parser = argparse.ArgumentParser(
    description="Run a model, including visualization and plotting.")
parser.add_argument("-p", "--model_path", type=str, default="./trained_models/apex",
                    help="File path for model to test")
parser.add_argument("-x", "--no-visualize", dest="visualize", default=True, action='store_false',
                    help="Don't render the policy.")
# parser.add_argument("--viz-target", default=False, action='store_true',
#                     help="Length of trajectory to visualize")
# parser.add_argument("-g", "--graph", dest="plot", default=False, action='store_true',
#                     help="Graph the output of the policy.")
# parser.add_argument("--glen", type=int, default=150,
#                     help="Length of trajectory to graph.")
parser.add_argument("--vlen", type=int, default=75,
                    help="Length of trajectory to visualize")
# parser.add_argument("--noise", default=False, action="store_true",
#                     help="Visualize policy with exploration.")
parser.add_argument('--env_name', default="Cassie-mimic-walking-v0",
                    help='name of the environment to run')
parser.add_argument("--state_est", type=bool, default=True)     # use state estimator or not

# parser.add_argument('--algo_name', default="TD3",
#                     help='name of the algo model to load')
args = parser.parse_args()

# create visdom logger
# plotter = VisdomLinePlotter(env_name=args.env_name)

# Environment
if(args.env_name in ["Cassie-v0", "Cassie-mimic-v0", "Cassie-mimic-walking-v0"]):
    # NOTE: importing cassie for some reason breaks openai gym, BUG ?
    from cassie import CassieEnv, CassieTSEnv
    from cassie.no_delta_env import CassieEnv_nodelta
    from cassie.speed_env import CassieEnv_speed
    from cassie.speed_double_freq_env import CassieEnv_speed_dfreq
    from cassie.speed_no_delta_env import CassieEnv_speed_no_delta
    # set up cassie environment
    # import gym_cassie
    # env_fn = gym_factory(args.env_name)
    #env_fn = make_env_fn(state_est=args.state_est)
    #env_fn = functools.partial(CassieEnv_speed_dfreq, "walking", clock_based = True, state_est=args.state_est)
    env_fn = functools.partial(CassieTSEnv)
    obs_dim = env_fn().observation_space.shape[0]
    action_dim = env_fn().action_space.shape[0]
    if args.state_est:
        # with state estimator
        env_fn = functools.partial(SymmetricEnv, env_fn, mirrored_obs=[0, 1, 2, 3, 4, -10, -11, 12, 13, 14, -5, -6, 7, 8, 9, 15, 16, 17, 18, 19, 20, -26, -27, 28, 29, 30, -21, -22, 23, 24, 25, 31, 32, 33, 37, 38, 39, 34, 35, 36, 43, 44, 45, 40, 41, 42, 46, 47, 48], mirrored_act=[0,1,2,3,4,5,6,7,8,9])
    else:
        # without state estimator
        env_fn = functools.partial(SymmetricEnv, env_fn, mirrored_obs=[0, 1, 2, 3, 4, 5, -13, -14, 15, 16, 17,
                                        18, 19, -6, -7, 8, 9, 10, 11, 12, 20, 21, 22, 23, 24, 25, -33,
                                        -34, 35, 36, 37, 38, 39, -26, -27, 28, 29, 30, 31, 32, 40, 41, 42],
                                        mirrored_act = [0,1,2,3,4,5,6,7,8,9])
    max_episode_steps = 400
else:
    import gym
    env_fn = gym_factory(args.env_name)
    #max_episode_steps = env_fn()._max_episode_steps
    obs_dim = env_fn().observation_space.shape[0]
    action_dim = env_fn().action_space.shape[0]
    max_episode_steps = 1000    # For Humanoid-v2

state_dim = env_fn().observation_space.shape[0]
action_dim = env_fn().action_space.shape[0]
#max_action = float(env_fn().action_space.high[0])
max_action = 1
min_action = -1


# Load Policy
actor = Actor(state_dim, action_dim, max_action, 256, 256).to(device)
actor_path = os.path.join(args.model_path, "global_policy.pt")
print('Loading model from {}'.format(actor_path))
if actor_path is not None:
    actor.load_state_dict(torch.load(actor_path))
    actor.eval()

#evaluator_id = evaluator.remote(env_fn, actor, max_episode_steps)

if(not args.visualize):
    visualize(env_fn, actor, args.vlen)
