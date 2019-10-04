import gym
import torch

def print_logo(subtitle="", option=2):
  class color:
   BOLD   = '\033[1m\033[48m'
   END    = '\033[0m'
   ORANGE = '\033[38;5;202m'
   BLACK  = '\033[38;5;240m'

  print(color.BOLD + color.ORANGE +  "         .8.         " + color.BLACK + " 8 888888888o   " + color.ORANGE + "8 8888888888   `8.`8888.      ,8' ")
  print(color.BOLD + color.ORANGE +  "        .888.        " + color.BLACK + " 8 8888    `88. " + color.ORANGE + "8 8888          `8.`8888.    ,8' ")
  print(color.BOLD + color.ORANGE +  "       :88888.       " + color.BLACK + " 8 8888     `88 " + color.ORANGE + "8 8888           `8.`8888.  ,8' ")
  print(color.BOLD + color.ORANGE +  "      . `88888.      " + color.BLACK + " 8 8888     ,88 " + color.ORANGE + "8 8888            `8.`8888.,8' ")
  print(color.BOLD + color.ORANGE +  "     .8. `88888.     " + color.BLACK + " 8 8888.   ,88' " + color.ORANGE + "8 888888888888     `8.`88888' ")
  print(color.BOLD + color.ORANGE + "    .8`8. `88888.    " + color.BLACK  + " 8 888888888P'  " + color.ORANGE + "8 8888             .88.`8888. ")
  print(color.BOLD + color.ORANGE + "   .8' `8. `88888.   " + color.BLACK  + " 8 8888         " + color.ORANGE + "8 8888            .8'`8.`8888. ")
  print(color.BOLD + color.ORANGE + "  .8'   `8. `88888.  " + color.BLACK  + " 8 8888         " + color.ORANGE + "8 8888           .8'  `8.`8888. ")
  print(color.BOLD + color.ORANGE + " .888888888. `88888. " + color.BLACK  + " 8 8888         " + color.ORANGE + "8 8888          .8'    `8.`8888. ")
  print(color.BOLD + color.ORANGE + ".8'       `8. `88888." + color.BLACK  + " 8 8888         " + color.ORANGE + "8 888888888888 .8'      `8.`8888. " + color.END)
  print("\n")
  print(subtitle)
  print("\n")

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

if __name__ == "__main__":
  import sys, argparse
  parser = argparse.ArgumentParser()

  print_logo(subtitle="Maintained by Oregon State University's Dynamic Robotics Lab")

  if len(sys.argv) < 2:
    print("Only got", sys.argv)

  elif sys.argv[1] == 'ars':
    """
      Utility for running Augmented Random Search.

    """
    from rl.algos.ars import run_experiment
    sys.argv.remove(sys.argv[1])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--env_name",     "-e",   default="Hopper-v2")
    parser.add_argument("--hidden_size",          default=32, type=int)
    parser.add_argument("--seed",         "-s",   default=0, type=int)
    parser.add_argument("--timesteps",    "-t",   default=1e8, type=int)
    parser.add_argument("--load_model",   "-l",   default=None, type=str)
    parser.add_argument("--save_model",   "-m",   default="./trained_models/ars/ars.pt", type=str)
    parser.add_argument('--std',          "-sd",  default=0.0075, type=float)
    parser.add_argument("--deltas",       "-d",   default=64, type=int)
    parser.add_argument("--lr",           "-lr",  default=0.01, type=float)
    parser.add_argument("--reward_shift", "-rs",  default=1, type=float)
    parser.add_argument("--traj_len",     "-tl",  default=1000, type=int)
    parser.add_argument("--algo",         "-a",   default='v1', type=str)
    parser.add_argument("--recurrent",    "-r",   action='store_true')
    parser.add_argument("--log_dir",       default="./logs/ars/experiments/", type=str)
    parser.add_argument("--average_every", default=10, type=int)
    args = parser.parse_args()

    run_experiment(args)

  elif sys.argv[1] == 'ddpg':
    sys.argv.remove(sys.argv[1])
    """
      Utility for running Deep Deterministic Policy Gradients.

    """
    from rl.algos.ddpg import run_experiment
    sys.argv.remove(sys.argv[1])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--env_name",     "-e",   default="Hopper-v2")
    parser.add_argument("--hidden_size",          default=32, type=int)
    parser.add_argument("--seed",         "-s",   default=0, type=int)
    parser.add_argument("--timesteps",    "-t",   default=1e8, type=int)
    parser.add_argument("--load_model",   "-l",   default=None, type=str)
    parser.add_argument("--save_model",   "-m",   default="./trained_models/ars/ars.pt", type=str)
    parser.add_argument('--tau',                  default=0.01, type=float)
    parser.add_argument("--actor_lr",     "-alr", default=0.001, type=float)
    parser.add_argument("--critic_lr",    "-clr", default=0.005, type=float)
    parser.add_argument("--traj_len",     "-tl",  default=1000, type=int)
    parser.add_argument("--recurrent",    "-r",   action='store_true')
    parser.add_argument("--log_dir",       default="./logs/ddpg/experiments/", type=str)
    parser.add_argument("--average_every", default=10, type=int)
    args = parser.parse_args()

    run_experiment(args)

  elif sys.argv[1] == 'rdpg':
    sys.argv.remove(sys.argv[1])
    """
      Utility for running Recurrent Deterministic Policy Gradients.

    """
    raise NotImplementedError
    import rl.algos.rdpg
  elif sys.argv[1] == 'td3_sync':
    sys.argv.remove(sys.argv[1])
    """
      Utility for running Twin-Delayed Deep Deterministic policy gradients.

    """
    raise NotImplementedError
  elif sys.argv[1] == 'td3_async':
    sys.argv.remove(sys.argv[1])
    """
      Utility for running Twin-Delayed Deep Deterministic policy gradients (asynchronous).

    """
    raise NotImplementedError
  elif sys.argv[1] == 'ppo':
    sys.argv.remove(sys.argv[1])
    """
      Utility for running Proximal Policy Optimization.

    """
    raise NotImplementedError
  else:
    print("Invalid argument '{}'".format(sys.argv[1]))
