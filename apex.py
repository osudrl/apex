import gym
import torch
import hashlib, os
from collections import OrderedDict

class color:
 BOLD   = '\033[1m\033[48m'
 END    = '\033[0m'
 ORANGE = '\033[38;5;202m'
 BLACK  = '\033[38;5;240m'


def print_logo(subtitle="", option=2):
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

def create_logger(args):
  from torch.utils.tensorboard import SummaryWriter
  """Use hyperparms to set a directory to output diagnostic files."""

  arg_dict = args.__dict__
  assert "seed" in arg_dict, \
    "You must provide a 'seed' key in your command line arguments"
  assert "logdir" in arg_dict, \
    "You must provide a 'logdir' key in your command line arguments."

  # sort the keys so the same hyperparameters will always have the same hash
  arg_dict = OrderedDict(sorted(arg_dict.items(), key=lambda t: t[0]))

  # remove seed so it doesn't get hashed, store value for filename
  # same for logging directory
  seed = str(arg_dict.pop("seed"))
  logdir = str(arg_dict.pop('logdir'))

  # get a unique hash for the hyperparameter settings, truncated at 10 chars
  arg_hash = hashlib.md5(str(arg_dict).encode('ascii')).hexdigest()[0:10] + '-seed' + seed
  output_dir = os.path.join(logdir, arg_hash)

  # create a directory with the hyperparm hash as its name, if it doesn't
  # already exist.
  os.makedirs(output_dir, exist_ok=True)

  # Create a file with all the hyperparam settings in plaintext
  info_path = os.path.join(output_dir, "experiment.info")
  file = open(info_path, 'w')
  for key, val in arg_dict.items():
      file.write("%s: %s" % (key, val))
      file.write('\n')

  logger = SummaryWriter(output_dir, flush_secs=0.1)
  print("Logging to " + color.BOLD + color.ORANGE + str(output_dir) + color.END)
  return logger

def eval_policy(policy, max_traj_len=1000, visualize=True, env_name=None):
  if env_name is None:
    env = gym_factory(policy.env_name)()
  else:
    env = gym_factory(env_name)()

  while True:
    state = env.reset()
    done = False
    timesteps = 0
    eval_reward = 0
    while not done and timesteps < 1000:
      action = policy.forward(torch.Tensor(state)).detach().numpy()
      state, reward, done, _ = env.step(action)
      if visualize:
        env.render()
      eval_reward += reward
      timesteps += 1
    print("Eval reward: ", eval_reward)

if __name__ == "__main__":
  import sys, argparse
  parser = argparse.ArgumentParser()

  print_logo(subtitle="Maintained by Oregon State University's Dynamic Robotics Lab")

  if len(sys.argv) < 2:
    print("Usage: python apex.py [algorithm name]", sys.argv)

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
    parser.add_argument("--logdir",       default="./logs/ars/experiments/", type=str)
    parser.add_argument("--average_every", default=10, type=int)
    args = parser.parse_args()

    run_experiment(args)

  elif sys.argv[1] == 'ddpg':
    sys.argv.remove(sys.argv[1])
    """
      Utility for running Deep Deterministic Policy Gradients.

    """
    from rl.algos.ddpg import run_experiment
    parser.add_argument("--workers",                default=1, type=int)
    parser.add_argument("--env_name",        "-e",  default="Hopper-v2")
    parser.add_argument("--hidden_size",            default=300, type=int)
    parser.add_argument("--seed",            "-s",  default=0, type=int)
    parser.add_argument("--timesteps",       "-t",  default=1e6, type=int)
    parser.add_argument("--start_timesteps",        default=1e4, type=int)
    parser.add_argument("--load_actor",             default=None, type=str)
    parser.add_argument("--load_critic",            default=None, type=str)
    parser.add_argument("--save_actor",             default="./trained_models/ddpg/ddpg_actor.pt", type=str)
    parser.add_argument("--save_critic",            default="./trained_models/ddpg/ddpg_critic.pt", type=str)
    parser.add_argument('--discount',               default=0.99, type=float)
    parser.add_argument('--expl_noise',             default=0.2, type=float)
    parser.add_argument('--tau',                    default=0.001, type=float)
    parser.add_argument("--actor_lr",       "-alr", default=5e-5, type=float)
    parser.add_argument("--critic_lr",      "-clr", default=5e-4, type=float)
    parser.add_argument("--traj_len",       "-tl",  default=1000, type=int)
    parser.add_argument("--center_reward",  "-r",   action='store_true')
    parser.add_argument("--batch_size",             default=64, type=int)
    parser.add_argument("--logdir",                 default="./logs/ddpg/experiments/", type=str)
    parser.add_argument("--eval_every",             default=100, type=int)
    args = parser.parse_args()

    run_experiment(args)

  elif sys.argv[1] == 'rdpg':
    sys.argv.remove(sys.argv[1])
    """
        Utility for running Recurrent Deterministic Policy Gradients.

    """
    from rl.algos.rdpg import run_experiment
    parser.add_argument("--workers",                default=1, type=int)
    parser.add_argument("--env_name",        "-e",  default="Hopper-v2")
    parser.add_argument("--hidden_size",            default=300, type=int)
    parser.add_argument("--seed",            "-s",  default=0, type=int)
    parser.add_argument("--timesteps",       "-t",  default=1e6, type=int)
    parser.add_argument("--start_timesteps",        default=1e4, type=int)
    parser.add_argument("--load_model",      "-l",  default=None, type=str)
    parser.add_argument("--save_model",      "-m",  default="./trained_models/ddpg/ddpg.pt", type=str)
    parser.add_argument('--discount',               default=0.99, type=float)
    parser.add_argument('--expl_noise',             default=0.2, type=float)
    parser.add_argument('--tau',                    default=0.001, type=float)
    parser.add_argument("--actor_lr",       "-alr", default=5e-5, type=float)
    parser.add_argument("--critic_lr",      "-clr", default=5e-4, type=float)
    parser.add_argument("--traj_len",       "-tl",  default=1000, type=int)
    parser.add_argument("--center_reward",  "-r",   action='store_true')
    parser.add_argument("--batch_size",             default=64, type=int)
    parser.add_argument("--logdir",                 default="./logs/ddpg/experiments/", type=str)
    parser.add_argument("--eval_every",             default=100, type=int)
    args = parser.parse_args()

    run_experiment(args)
  elif sys.argv[1] == 'td3_sync':
    sys.argv.remove(sys.argv[1])
    """
      Utility for running Twin-Delayed Deep Deterministic policy gradients.

    """
    from rl.algos.sync_td3 import run_experiment
    parser.add_argument("--logdir",       default="./logs/syncTD3/experiments/", type=str)
    parser.add_argument("--policy_name", default="TD3")					            # Policy name
    parser.add_argument("--num_procs", type=int, default=4)                         # neurons in hidden layer
    parser.add_argument("--min_steps", type=int, default=1000)                      # number of steps of experience each process should collect
    parser.add_argument("--max_traj_len", type=int, default=400)                    # max steps in each episode
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
    parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G')      # initial scale of noise for param noise
    parser.add_argument("--batch_size", default=100, type=int)                      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)                     # Discount factor
    parser.add_argument("--tau", default=0.001, type=float)                         # Target network update rate
    parser.add_argument("--a_lr", type=float, default=3e-4)                         # Actor: Adam learning rate
    parser.add_argument("--c_lr", type=float, default=1e-3)                         # Critic: Adam learning rate
    # TD3 Specific
    parser.add_argument("--policy_noise", default=0.2, type=float)                  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)                    # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)                       # Frequency of delayed policy updates
    args = parser.parse_args()

    run_experiment(args)
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
  elif sys.argv[1] == 'eval':
    sys.argv.remove(sys.argv[1])

    parser.add_argument("--policy", default="./trained_models/ddpg/ddpg_actor.pt", type=str)
    args = parser.parse_args()

    policy = torch.load(args.policy)

    eval_policy(policy)
  else:
    print("Invalid algorithm '{}'".format(sys.argv[1]))
