import os
import time
import torch
import numpy as np

from cassie import CassieEnv, CassieTrajEnv, CassiePlayground, CassieStandingEnv

def env_factory(path, command_profile="clock", input_profile="full", simrate=50, dynamics_randomization=True, mirror=False, learn_gains=False, reward=None, history=0, no_delta=True, traj=None, ik_baseline=False, **kwargs):
    from functools import partial

    """
    Returns an *uninstantiated* environment constructor.

    Since environments containing cpointers (e.g. Mujoco envs) can't be serialized,
    this allows us to pass their constructors to Ray remote functions instead
    (since the gym registry isn't shared across ray subprocesses we can't simply
    pass gym.make() either)

    Note: env.unwrapped.spec is never set, if that matters for some reason.
    """

    # Custom Cassie Environment
    if path in ['Cassie-v0', 'CassieTraj-v0', 'CassiePlayground-v0', 'CassieStandingEnv-v0']:

        if path == 'Cassie-v0':
            env_fn = partial(CassieEnv, command_profile=command_profile, input_profile=input_profile, simrate=simrate, dynamics_randomization=dynamics_randomization, learn_gains=learn_gains, reward=reward, history=history)
        elif path == 'CassieTraj-v0':
            env_fn = partial(CassieTrajEnv, traj=traj, command_profile=command_profile, input_profile=input_profile, simrate=simrate, dynamics_randomization=dynamics_randomization, no_delta=no_delta, learn_gains=learn_gains, ik_baseline=ik_baseline, reward=reward, history=history)
        elif path == 'CassiePlayground-v0':
            env_fn = partial(CassiePlayground, command_profile=command_profile, input_profile=input_profile, simrate=simrate, dynamics_randomization=dynamics_randomization, learn_gains=learn_gains, reward=reward, history=history)
        elif path == 'CassieStandingEnv-v0':
            env_fn = partial(CassieStandingEnv, command_profile=command_profile, input_profile=input_profile, simrate=simrate, dynamics_randomization=dynamics_randomization, learn_gains=learn_gains, reward=reward, history=history)

        if mirror:
            from rl.envs.wrappers import SymmetricEnv
            env_fn = partial(SymmetricEnv, env_fn, mirrored_obs=env_fn().mirrored_obs, mirrored_act=env_fn().mirrored_acts)

        print()
        print("Environment: {}".format(path))
        print(" ├ reward:         {}".format(reward))
        print(" ├ input prof:     {}".format(input_profile))
        print(" ├ cmd prof:       {}".format(command_profile))
        print(" ├ learn gains:    {}".format(learn_gains))
        print(" ├ dyn_random:     {}".format(dynamics_randomization))
        print(" ├ mirror:         {}".format(mirror))
        if path == "CassieTraj-v0":
            print(" ├ traj:           {}".format(traj))
            print(" ├ ik baseline:    {}".format(ik_baseline))
            print(" ├ no_delta:       {}".format(no_delta))
        print(" └ obs_dim:        {}".format(env_fn().observation_space.shape[0]))

        return env_fn

    # OpenAI Gym environment
    else:
        import gym
        spec = gym.envs.registry.spec(path)
        _kwargs = spec._kwargs.copy()
        _kwargs.update(kwargs)

        try:
            if callable(spec._entry_point):
                cls = spec._entry_point(**_kwargs)
            else:
                cls = gym.envs.registration.load(spec._entry_point)
        except AttributeError:
            if callable(spec.entry_point):
                cls = spec.entry_point(**_kwargs)
            else:
                cls = gym.envs.registration.load(spec.entry_point)

        return partial(cls, **_kwargs)
