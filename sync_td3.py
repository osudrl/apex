import numpy as np
import torch

import argparse
import os

from apex import print_logo

from rl.utils import ReplayBuffer, AdaptiveParamNoiseSpec, distance_metric
from rl.algos.sync_td3 import TD3, parallel_collect_experience

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from colorama import Fore, Style

import gym

import functools

import ray

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

# Runs policy for X episodes and returns average reward. Optionally render policy
def evaluate_policy(env, policy, eval_episodes=1):
    avg_reward = 0.0
    avg_eplen = 0.0
    for _ in range(eval_episodes):
        obs = env.reset()
        t = 0
        done_bool = 0.0
        while not done_bool:
            t += 1
            action = policy.select_action(np.array(obs), param_noise=None)
            obs, reward, done, _ = env.step(action)
            done_bool = 1.0 if t + 1 == max_traj_len else float(done)
            avg_reward += reward
        avg_eplen += t

    avg_reward /= eval_episodes
    avg_eplen /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward, avg_eplen

if __name__ == "__main__":

    # General
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis_address", type=str, default=None)                  # address of redis server (for cluster setups)
    parser.add_argument("--policy_name", default="TD3")					            # Policy name
    parser.add_argument("--num_procs", type=int, default=4)                         # neurons in hidden layer
    parser.add_argument("--min_steps", type=int, default=1000)                      # number of steps of experience each process should collect
    parser.add_argument("--max_traj_len", type=int, default=400)                      # max steps in each episode

    parser.add_argument("--env_name", default="Cassie-mimic-v0")                    # environment name
    parser.add_argument("--hidden_size", default=256)                               # neurons in hidden layer
    parser.add_argument("--state_est", default=True, action='store_true')           # use state estimator or not
    parser.add_argument("--mirror", default=False, action='store_true')             # mirror actions or not

    parser.add_argument("--seed", default=0, type=int)                              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)                 # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)                     # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e7, type=float)                 # Max time steps to run environment for
    parser.add_argument("--save_models", default=True, action="store_true")         # Whether or not models are saved
    
    parser.add_argument("--act_noise", default=0.3, type=float)                     # Std of Gaussian exploration noise (used to be 0.1)
    parser.add_argument('--param_noise', type=bool, default=False)                  # param noise
    parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                        help='initial param noise scale (default: 0.3)')

    parser.add_argument("--batch_size", default=100, type=int)                      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)                     # Discount factor
    parser.add_argument("--tau", default=0.001, type=float)                         # Target network update rate
    parser.add_argument("--a_lr", type=float, default=3e-4)                         # Actor: Adam learning rate
    parser.add_argument("--c_lr", type=float, default=1e-3)                         # Critic: Adam learning rate

    # TD3 Specific
    parser.add_argument("--policy_noise", default=0.2, type=float)                  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)                    # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)                       # Frequency of delayed policy updates

    # For tensorboard logger
    parser.add_argument("--logdir", type=str, default="./logs/synctd3/experiments/")       # Where to log diagnostics to

    args = parser.parse_args()

    print_logo(subtitle="Distributed Twin-Delayed DDPG")

    ray.init(num_gpus=0, include_webui=True, temp_dir="./ray_tmp", redis_address=args.redis_address)

    file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")

    # Tensorboard logging
    now = datetime.now()
    # NOTE: separate by trial name first and time of run after
    log_path = args.logdir + now.strftime("%Y%m%d-%H%M%S")+"/"
    logger = SummaryWriter(log_path, flush_secs=0.1)
    print(Fore.GREEN + Style.BRIGHT + "Logging data using TensorBoard to {}".format(log_path + Style.RESET_ALL))

    print(args.env_name)

    # BAD practice??
    global max_episode_steps

    # Environment
    if(args.env_name in ["Cassie-v0", "Cassie-mimic-v0", "Cassie-mimic-walking-v0"]):
        cassieEnv = True
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
        max_traj_len = args.max_traj_len
    else:
        cassieEnv = False
        import gym
        env_fn = gym_factory(args.env_name)
        #max_episode_steps = env_fn()._max_episode_steps
        obs_dim = env_fn().observation_space.shape[0]
        action_dim = env_fn().action_space.shape[0]
        max_traj_len = 1000

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env_fn().observation_space.shape[0]
    action_dim = env_fn().action_space.shape[0]
    max_action = 1.0
    #max_action = float(env.action_space.high[0])

    print("state_dim: {}".format(state_dim))
    print("action_dim: {}".format(action_dim))
    print("max_action dim: {}".format(max_action))
    print("max_episode_steps: {}".format(max_traj_len))

    # Initialize policy
    policy = TD3(state_dim, action_dim, max_action, a_lr=args.a_lr, c_lr=args.c_lr)

    replay_buffer = ReplayBuffer()

    # Initialize param noise (or set to None)
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05, desired_action_stddev=args.noise_scale, adaptation_coefficient=1.05) if args.param_noise else None

    total_timesteps = 0
    total_updates = 0
    timesteps_since_eval = 0
    episode_num = 0
    
    # Evaluate untrained policy
    ret, eplen = evaluate_policy(env_fn(), policy)
    logger.add_scalar("Eval/Return", ret, total_updates)
    logger.add_scalar("Eval/Eplen", eplen, total_updates)

    while total_timesteps < args.max_timesteps:

        # collect parallel experience and add to replay buffer
        merged_transitions, episode_timesteps = parallel_collect_experience(policy, env_fn, args.act_noise, args.min_steps, max_traj_len, num_procs=args.num_procs)
        replay_buffer.add_parallel(merged_transitions)
        total_timesteps += episode_timesteps
        timesteps_since_eval += episode_timesteps
        episode_num += args.num_procs

        # Logging rollouts
        print("Total T: {} Episode Num: {} Episode T: {}".format(total_timesteps, episode_num, episode_timesteps))

        # update the policy
        avg_q1, avg_q2, avg_targ_q, q_loss, pi_loss, avg_action = policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau, args.policy_noise, args.noise_clip, args.policy_freq)
        total_updates += episode_timesteps      # this is how many iterations we did updates for

        # Logging training
        logger.add_scalar("Train/avg_q1", avg_q1, total_updates)
        logger.add_scalar("Train/avg_q2", avg_q2, total_updates)
        logger.add_scalar("Train/avg_targ_q", avg_targ_q, total_updates)
        logger.add_scalar("Train/q_loss", q_loss, total_updates)
        logger.add_scalar("Train/pi_loss", pi_loss, total_updates)
        logger.add_histogram("Train/avg_action", avg_action, total_updates)

        # Evaluate episode
        if timesteps_since_eval >= args.eval_freq:
            timesteps_since_eval = 0
            ret, eplen = evaluate_policy(env_fn(), policy)

            # Logging Eval
            logger.add_scalar("Eval/Return", ret, total_updates)
            logger.add_scalar("Eval/Eplen", eplen, total_updates)
            logger.add_histogram("Eval/avg_action", avg_action, total_updates)

            # Logging Totals
            logger.add_scalar("Total/Timesteps", total_timesteps, total_updates)
            logger.add_scalar("Total/ReplaySize", replay_buffer.ptr, total_updates)

            if args.save_models:
                policy.save()

    # Final evaluation
    ret, eplen = evaluate_policy(env_fn(), policy)
    logger.add_scalar("Eval/Return", ret, total_updates)
    logger.add_scalar("Eval/Eplen", eplen, total_updates)

    # Final Policy Save
    if args.save_models:
        policy.save()