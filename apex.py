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
  print()
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

def env_factory(path, state_est=True, mirror=False, speed=None, clock_based=False, **kwargs):
    from functools import partial

    """
    Returns an *uninstantiated* environment constructor.

    Since environments containing cpointers (e.g. Mujoco envs) can't be serialized, 
    this allows us to pass their constructors to Ray remote functions instead 
    (since the gym registry isn't shared across ray subprocesses we can't simply 
    pass gym.make() either)

    Note: env.unwrapped.spec is never set, if that matters for some reason.
    """
    if path in ['Cassie-v0', 'CassieMimic-v0', 'CassieRandomDynamics-v0', 'CassieIK-v0', 'CassieIKNoDelta-v0', 'CassieStandingEnv-v0', 'CassieGroundFrictionEnv-v0']:
      from cassie import CassieEnv, CassieTSEnv, CassieIKEnv, UnifiedCassieIKEnv, UnifiedCassieIKEnvNoDelta, CassieEnv_nodelta, CassieEnv_rand_dyn, CassieEnv_speed_dfreq, CassieStandingEnv, CassieGroundFrictionEnv

      if path == 'Cassie-v0':
        env_fn = partial(CassieEnv, "walking", clock_based=clock_based, state_est=state_est)
      elif path == 'CassieRandomDynamics-v0':
        env_fn = partial(CassieEnv_rand_dyn, "walking", clock_based=clock_based, state_est=state_est)
      elif path == 'CassieRandomDynamics-v0':
        env_fn = partial(CassieEnv_rand_dyn, "walking", clock_based=True, state_est=False)
      elif path == 'CassieIK-v0':
        # env_fn = partial(CassieIKEnv, "walking", clock_based=True, state_est=state_est, speed=speed)
        env_fn = partial(UnifiedCassieIKEnv, "walking", clock_based=True, state_est=state_est)
      elif path == 'CassieIKNoDelta-v0':
        env_fn = partial(UnifiedCassieIKEnvNoDelta, "walking", clock_based=True, state_est=state_est)
      elif path == 'CassieStandingEnv-v0':
        env_fn = partial(CassieStandingEnv, "stepping", simrate=60, state_est=True)
      elif path == 'CassieGroundFrictionEnv-v0':
        env_fn = partial(CassieGroundFrictionEnv, "walking", clock_based=True, state_est=True)


      if mirror:
          from rl.envs.wrappers import SymmetricEnv
          if state_est:
              # with state estimator
              env_fn = partial(SymmetricEnv, env_fn, mirrored_obs=[0.1, 1, 2, 3, 4, -10, -11, 12, 13, 14, -5, -6, 7, 8, 9, 15, 16, 17, 18, 19, 20, -26, -27, 28, 29, 30, -21, -22, 23, 24, 25, 31, 32, 33, 37, 38, 39, 34, 35, 36, 43, 44, 45, 40, 41, 42, 46, 47], mirrored_act=[-5, -6, 7, 8, 9, -0.1, -1, 2, 3, 4])
          else:
              # without state estimator
              env_fn = partial(SymmetricEnv, env_fn, mirrored_obs=[0.1, 1, 2, 3, 4, 5, -13, -14, 15, 16, 17,
                                              18, 19, -6, -7, 8, 9, 10, 11, 12, 20, 21, 22, 23, 24, 25, -33,
                                              -34, 35, 36, 37, 38, 39, -26, -27, 28, 29, 30, 31, 32, 40, 41, 42],
                                              mirrored_act = [-5, -6, 7, 8, 9, -0.1, -1, 2, 3, 4])

      return env_fn

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
  assert "env_name" in arg_dict, \
    "You must provide a 'env_name' key in your command line arguments."

  # sort the keys so the same hyperparameters will always have the same hash
  arg_dict = OrderedDict(sorted(arg_dict.items(), key=lambda t: t[0]))

  # remove seed so it doesn't get hashed, store value for filename
  # same for logging directory
  seed = str(arg_dict.pop("seed"))
  logdir = str(arg_dict.pop('logdir'))
  env_name = str(arg_dict.pop('env_name'))

  # get a unique hash for the hyperparameter settings, truncated at 10 chars
  arg_hash   = hashlib.md5(str(arg_dict).encode('ascii')).hexdigest()[0:6] + '-seed' + seed
  logdir     = os.path.join(logdir, env_name)
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

  logger.dir = output_dir
  return logger

def eval_policy(policy, max_traj_len=1000, visualize=True, env_name=None, speed=0.0, state_est=True, clock_based=False):

  if env_name is None:
    env = env_factory(policy.env_name, speed=speed, state_est=state_est, clock_based=clock_based)()
  else:
    env = env_factory(env_name, speed=speed, state_est=state_est, clock_based=clock_based)()

  while True:
    state = env.reset()
    done = False
    timesteps = 0
    eval_reward = 0
    while not done and timesteps < max_traj_len:

      if hasattr(env, 'simrate'):
        start = time.time()
      
      action = policy.forward(torch.Tensor(state)).detach().numpy()
      state, reward, done, _ = env.step(action)
      if visualize:
        env.render()
      eval_reward += reward
      timesteps += 1

      if hasattr(env, 'simrate'):
        # assume 30hz (hack)
        end = time.time()
        delaytime = max(0, 1000 / 30000 - (end-start))
        time.sleep(delaytime)

    print("Eval reward: ", eval_reward)

if __name__ == "__main__":
  import sys, argparse, time
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
    parser.add_argument("--hidden_size",          default=32, type=int)                 # neurons in hidden layer
    parser.add_argument("--timesteps",    "-t",   default=1e8, type=int)                # timesteps to run experiment ofr
    parser.add_argument("--load_model",   "-l",   default=None, type=str)               # load a model from a saved file.
    parser.add_argument('--std',          "-sd",  default=0.0075, type=float)           # the standard deviation of the parameter noise vectors
    parser.add_argument("--deltas",       "-d",   default=64, type=int)                 # number of parameter noise vectors to use
    parser.add_argument("--lr",           "-lr",  default=0.01, type=float)             # the learning rate used to update policy
    parser.add_argument("--reward_shift", "-rs",  default=1, type=float)                # the reward shift (to counter Gym's alive_bonus)
    parser.add_argument("--traj_len",     "-tl",  default=1000, type=int)               # max trajectory length for environment
    parser.add_argument("--algo",         "-a",   default='v1', type=str)               # whether to use ars v1 or v2
    parser.add_argument("--recurrent",    "-r",   action='store_true')                  # whether to use a recurrent policy
    parser.add_argument("--logdir",               default="./logs/ars/", type=str)
    parser.add_argument("--seed",     "-s",       default=0, type=int)
    parser.add_argument("--env_name", "-e",       default="Hopper-v3")
    parser.add_argument("--average_every",        default=10, type=int)
    parser.add_argument("--save_model",   "-m",   default=None, type=str)               # where to save the trained model to
    parser.add_argument("--redis",                default=None)
    args = parser.parse_args()

    run_experiment(args)

  elif sys.argv[1] == 'ddpg' or sys.argv[1] == 'rdpg':

    if sys.argv[1] == 'ddpg':
      recurrent = False
    if sys.argv[1] == 'rdpg':
      recurrent = True

    sys.argv.remove(sys.argv[1])
    """
      Utility for running Recurrent/Deep Deterministic Policy Gradients.
    """
    from rl.algos.dpg import run_experiment
    parser.add_argument("--hidden_size",            default=32,   type=int)       # neurons in hidden layers
    parser.add_argument("--layers",                 default=2,     type=int)      # number of hidden layres
    parser.add_argument("--timesteps",       "-t",  default=1e6,   type=int)      # number of timesteps in replay buffer
    parser.add_argument("--start_timesteps",        default=1e4,   type=int)      # number of timesteps to generate random actions for
    parser.add_argument("--load_actor",             default=None,  type=str)      # load an actor from a .pt file
    parser.add_argument("--load_critic",            default=None,  type=str)      # load a critic from a .pt file
    parser.add_argument('--discount',               default=0.99,  type=float)    # the discount factor
    parser.add_argument('--expl_noise',             default=0.2,   type=float)    # random noise used for exploration
    parser.add_argument('--tau',                    default=0.01, type=float)     # update factor for target networks
    parser.add_argument("--a_lr",           "-alr", default=1e-5,  type=float)    # adam learning rate for critic
    parser.add_argument("--c_lr",           "-clr", default=1e-4,  type=float)    # adam learning rate for actor
    parser.add_argument("--traj_len",       "-tl",  default=1000,  type=int)      # max trajectory length for environment
    parser.add_argument("--center_reward",  "-r",   action='store_true')          # normalize rewards to a normal distribution
    parser.add_argument("--normalize",              action='store_true')          # normalize states using welford's algorithm
    parser.add_argument("--batch_size",             default=64,    type=int)      # batch size for policy update
    parser.add_argument("--updates",                default=1,    type=int)       # (if recurrent) number of times to update policy per episode
    parser.add_argument("--eval_every",             default=100,   type=int)      # how often to evaluate the trained policy
    parser.add_argument("--save_actor",             default=None, type=str)
    parser.add_argument("--save_critic",            default=None, type=str)

    if not recurrent:
      parser.add_argument("--logdir",                 default="./logs/ddpg/", type=str)
    else:
      parser.add_argument("--logdir",                 default="./logs/rdpg/", type=str)

    parser.add_argument("--seed",     "-s",   default=0, type=int)
    parser.add_argument("--env_name", "-e",   default="Hopper-v3")
    args = parser.parse_args()

    args.recurrent = recurrent

    run_experiment(args)

  elif sys.argv[1] == 'td3_sync':
    sys.argv.remove(sys.argv[1])
    """
      Utility for running Twin-Delayed Deep Deterministic policy gradients.

    """
    from rl.algos.sync_td3 import run_experiment

    # general args
    parser.add_argument("--logdir",       default="./logs/syncTD3/experiments/", type=str)
    parser.add_argument("--policy_name", default="TD3")					            # Policy name
    parser.add_argument("--env_name", default="Cassie-v0")                    # environment name
    parser.add_argument("--state_est", default=True, action='store_true')           # use state estimator or not
    parser.add_argument("--mirror", default=False, action='store_true')             # mirror actions or not
    parser.add_argument("--redis_address", type=str, default=None)                  # address of redis server (for cluster setups)
    parser.add_argument("--seed", default=0, type=int)                              # Sets Gym, PyTorch and Numpy seeds

    # DDPG args
    parser.add_argument("--num_procs", type=int, default=4)                         # neurons in hidden layer
    parser.add_argument("--min_steps", type=int, default=1000)                      # number of steps of experience each process should collect
    parser.add_argument("--max_traj_len", type=int, default=400)                    # max steps in each episode
    parser.add_argument("--hidden_size", default=256)                               # neurons in hidden layer
    parser.add_argument("--start_timesteps", default=1e4, type=int)                 # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e4, type=float)                     # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e7, type=float)                 # Max time steps to run environment for
    parser.add_argument("--save_models", default=True, action="store_true")         # Whether or not models are saved
    parser.add_argument("--act_noise", default=0.3, type=float)                     # Std of Gaussian exploration noise (used to be 0.1)
    parser.add_argument('--param_noise', type=bool, default=False)                  # param noise
    parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G')      # initial scale of noise for param noise
    parser.add_argument("--batch_size", default=64, type=int)                       # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)                     # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)                         # Target network update rate
    parser.add_argument("--a_lr", type=float, default=1e-4)                         # Actor: Adam learning rate
    parser.add_argument("--c_lr", type=float, default=1e-4)                         # Critic: Adam learning rate

    # TD3 Specific
    parser.add_argument("--policy_noise", default=0.2, type=float)                  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)                    # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)                       # Frequency of delayed policy updates
    args = parser.parse_args()

    run_experiment(args)
  elif sys.argv[1] == 'asyncTD3':
    sys.argv.remove(sys.argv[1])
    """
      Utility for running Twin-Delayed Deep Deterministic policy gradients (asynchronous).

    """
    from rl.algos.async_td3 import run_experiment

    # args common for actors and learners
    parser.add_argument("--env_name", default="Cassie-v0")                    # environment name
    parser.add_argument("--hidden_size", default=256)                               # neurons in hidden layer
    parser.add_argument("--state_est", default=True, action='store_true')           # use state estimator or not
    parser.add_argument("--mirror", default=False, action='store_true')             # mirror actions or not

    # learner specific args
    parser.add_argument("--replay_size", default=1e8, type=int)                     # Max size of replay buffer
    parser.add_argument("--max_timesteps", default=1e8, type=float)                 # Max time steps to run environment for 1e8 == 100,000,000
    parser.add_argument("--batch_size", default=64, type=int)                      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)                     # exploration/exploitation discount factor
    parser.add_argument("--tau", default=0.005, type=float)                         # target update rate (tau)
    parser.add_argument("--update_freq", default=2, type=int)                      # how often to update learner
    parser.add_argument("--evaluate_freq", default=5000, type=int)                    # how often to evaluate learner
    parser.add_argument("--a_lr", type=float, default=3e-4)                         # Actor: Adam learning rate
    parser.add_argument("--c_lr", type=float, default=1e-4)                         # Critic: Adam learning rate

    # actor specific args
    parser.add_argument("--num_procs", default=30, type=int)                        # Number of actors
    parser.add_argument("--max_traj_len", type=int, default=400)                    # max steps in each episode
    parser.add_argument("--start_timesteps", default=1e4, type=int)                 # How many time steps purely random policy is run for
    parser.add_argument("--initial_load_freq", default=10, type=int)                # initial amount of time between loading global model
    parser.add_argument("--act_noise", default=0.3, type=float)                     # Std of Gaussian exploration noise (used to be 0.1)
    parser.add_argument('--param_noise', type=bool, default=False)                   # param noise
    parser.add_argument('--noise_scale', type=float, default=0.3)                   # noise scale for param noise
    parser.add_argument("--taper_load_freq", type=bool, default=True)               # taper the load frequency over the course of training or not
    parser.add_argument("--viz_actors", default=False, action='store_true')         # Visualize actors in visdom or not

    # evaluator args
    parser.add_argument("--num_trials", default=10, type=int)                       # Number of evaluators
    parser.add_argument("--num_evaluators", default=10, type=int)                   # Number of evaluators
    parser.add_argument("--viz_port", default=8097)                                 # visdom server port
    parser.add_argument("--render_policy", type=bool, default=False)                # render during eval

    # misc args
    parser.add_argument("--policy_name", type=str, default="model")                 # name to save policy to
    parser.add_argument("--seed", type=int, default=1, help="RNG seed")
    parser.add_argument("--logger_name", type=str, default="tensorboard")           # logger to use (tensorboard or visdom)
    parser.add_argument("--logdir", type=str, default="./logs/asynctd3/experiments/", help="Where to log diagnostics to")
    parser.add_argument("--redis_address", type=str, default=None)                  # address of redis server (for cluster setups)

    args = parser.parse_args()

    run_experiment(args)
  elif sys.argv[1] == 'ppo':
    sys.argv.remove(sys.argv[1])
    """
      Utility for running Proximal Policy Optimization.

    """
    from rl.algos.mirror_ppo import run_experiment

    # general args
    parser.add_argument("--policy_name", type=str, default="PPO")
    parser.add_argument("--env_name", "-e",   default="CassieIK-v0")
    parser.add_argument("--logdir", type=str, default="./logs/ppo/experiments/")        # Where to log diagnostics to
    parser.add_argument("--previous", type=str, default=None)                           # path to directory of previous policies for resuming training
    parser.add_argument("--seed", default=0, type=int)                                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--state_est", type=bool, default=True)                         # use state estimator or not
    parser.add_argument("--mirror", default=False, action='store_true')                 # mirror actions or not   
    parser.add_argument("--redis_address", type=str, default=None)                      # address of redis server (for cluster setups)
    parser.add_argument("--viz_port", default=8097)                                     # (deprecated) visdom server port

    # PPO algo args
    parser.add_argument("--input_norm_steps", type=int, default=10000)
    parser.add_argument("--n_itr", type=int, default=10000, help="Number of iterations of the learning algorithm")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate") # Xie
    parser.add_argument("--eps", type=float, default=1e-5, help="Adam epsilon (for numerical stability)")
    parser.add_argument("--lam", type=float, default=0.95, help="Generalized advantage estimate discount")
    parser.add_argument("--gamma", type=float, default=0.99, help="MDP discount")
    parser.add_argument("--entropy_coeff", type=float, default=0.0, help="Coefficient for entropy regularization")
    parser.add_argument("--clip", type=float, default=0.2, help="Clipping parameter for PPO surrogate loss")
    parser.add_argument("--minibatch_size", type=int, default=64, help="Batch size for PPO updates")
    parser.add_argument("--epochs", type=int, default=3, help="Number of optimization epochs per PPO update") #Xie
    parser.add_argument("--num_steps", type=int, default=5096, help="Number of sampled timesteps per gradient estimate")
    parser.add_argument("--use_gae", type=bool, default=True,help="Whether or not to calculate returns using Generalized Advantage Estimation")
    parser.add_argument("--num_procs", type=int, default=30, help="Number of threads to train on")
    parser.add_argument("--max_grad_norm", type=float, default=0.05, help="Value to clip gradients at.")
    parser.add_argument("--max_traj_len", type=int, default=400, help="Max episode horizon")

    # arg for training on aslipik_env
    parser.add_argument("--speed", type=float, default=0.0, help="Speed of aslip env")

    args = parser.parse_args()
    args.num_steps = args.num_steps // args.num_procs
    run_experiment(args)

  elif sys.argv[1] == 'eval':
    sys.argv.remove(sys.argv[1])

    parser.add_argument("--policy", default="./trained_models/ddpg/ddpg_actor.pt", type=str)
    parser.add_argument("--env_name", default="CassieIK-v0", type=str)
    parser.add_argument("--traj_len", default=400, type=str)
    parser.add_argument("--speed", type=float, default=0.0, help="Speed of aslip env")
    parser.add_argument("--state_est", default=True, action='store_true')           # use state estimator or not
    parser.add_argument("--clock_based", default=True, action='store_true')
    args = parser.parse_args()

    policy = torch.load(args.policy)


    eval_policy(policy, env_name=args.env_name, max_traj_len=args.traj_len, speed=args.speed, state_est=args.state_est, clock_based=args.clock_based)

    
  else:
    print("Invalid algorithm '{}'".format(sys.argv[1]))
