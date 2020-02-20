import torch
import hashlib, os, pickle
from collections import OrderedDict

from cassie.quaternion_function import *

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

def env_factory(path, traj="walking", clock_based=True, state_est=True, dynamics_randomization=True, mirror=False, no_delta=False, history=0, **kwargs):
    from functools import partial

    """
    Returns an *uninstantiated* environment constructor.

    Since environments containing cpointers (e.g. Mujoco envs) can't be serialized, 
    this allows us to pass their constructors to Ray remote functions instead 
    (since the gym registry isn't shared across ray subprocesses we can't simply 
    pass gym.make() either)

    Note: env.unwrapped.spec is never set, if that matters for some reason.
    """
    if history > 0:
      raise NotImplementedError

    # Custom Cassie Environment
    if path in ['Cassie-v0', 'CassieStandingEnv-v0']:
        from cassie import CassieEnv, CassieStandingEnv

        if path == 'Cassie-v0':
            env_fn = partial(CassieEnv, traj=traj, clock_based=clock_based, state_est=state_est, dynamics_randomization=dynamics_randomization, no_delta=no_delta, history=history)
        elif path == 'CassieStandingEnv-v0':
            env_fn = partial(CassieStandingEnv, state_est=state_est)

        # TODO for Yesh: make mirrored_obs an attribute of environment, configured based on setup parameters
        if mirror:
            from rl.envs.wrappers import SymmetricEnv
            env_fn = partial(SymmetricEnv, env_fn, mirrored_obs=env_fn().mirrored_obs, mirrored_act=[-5, -6, 7, 8, 9, -0.1, -1, 2, 3, 4])

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

# Logger stores in trained_models by default
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
    run_name = arg_dict.pop('run_name')
    seed = str(arg_dict.pop("seed"))
    logdir = str(arg_dict.pop('logdir'))
    env_name = str(arg_dict.pop('env_name'))

    # see if this run has a unique name, if so then that is going to be the name of the folder, even if it overrirdes
    if run_name is not None:
        logdir = os.path.join(logdir, env_name)
        output_dir = os.path.join(logdir, run_name)
    else:
        # see if we are resuming a previous run, if we are mark as continued
        if args.previous is not None:
            output_dir = args.previous[0:-1] + '-cont'
        else:
            # get a unique hash for the hyperparameter settings, truncated at 10 chars
            arg_hash   = hashlib.md5(str(arg_dict).encode('ascii')).hexdigest()[0:6] + '-seed' + seed
            logdir     = os.path.join(logdir, env_name)
            output_dir = os.path.join(logdir, arg_hash)

    # create a directory with the hyperparm hash as its name, if it doesn't
    # already exist.
    os.makedirs(output_dir, exist_ok=True)

    # Create a file with all the hyperparam settings in human-readable plaintext,
    # also pickle file for resuming training easily
    info_path = os.path.join(output_dir, "experiment.info")
    pkl_path = os.path.join(output_dir, "experiment.pkl")
    with open(pkl_path, 'wb') as file:
        pickle.dump(args, file)
    with open(info_path, 'w') as file:
        for key, val in arg_dict.items():
            file.write("%s: %s" % (key, val))
            file.write('\n')

    logger = SummaryWriter(output_dir, flush_secs=0.1) # flush_secs=0.1 actually slows down quite a bit, even on parallelized set ups
    print("Logging to " + color.BOLD + color.ORANGE + str(output_dir) + color.END)

    logger.dir = output_dir
    return logger


def eval_policy(policy, args, run_args):

    import tty
    import termios
    import select
    import numpy as np

    def isData():
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    max_traj_len = args.traj_len
    visualize = True

    env = env_factory(run_args.env_name, traj=run_args.traj, state_est=run_args.state_est, dynamics_randomization=run_args.dyn_random, mirror=run_args.mirror, clock_based=run_args.clock_based, history=run_args.history)()
    
    old_settings = termios.tcgetattr(sys.stdin)

    orient_add = 0

    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            
            state = env.reset()
            done = False
            timesteps = 0
            eval_reward = 0

            while not done and timesteps < max_traj_len:

                if isData():
                    c = sys.stdin.read(1)
                    if c == 'w':
                        env.speed += .1
                        print("speed: ", env.speed)
                    elif c == 's':
                        env.speed -= .1
                        print("speed: ", env.speed)
                    elif c == 'l':
                        orient_add += .1
                        print("Increasing orient_add to: ", orient_add)
                    elif c == 'k':
                        orient_add -= .1
                        print("Decreasing orient_add to: ", orient_add)
                    elif c == 'p':
                        push = 100
                        push_dir = 2
                        force_arr = np.zeros(6)
                        force_arr[push_dir] = push
                        env.sim.apply_force(force_arr)
                
                # Update Orientation
                quaternion = euler2quat(z=orient_add, y=0, x=0)
                iquaternion = inverse_quaternion(quaternion)

                if env.state_est:
                    curr_orient = state[1:5]
                    curr_transvel = state[14:17]
                else:
                    curr_orient = state[2:6]
                    curr_transvel = state[20:23]
                
                new_orient = quaternion_product(iquaternion, curr_orient)

                if new_orient[0] < 0:
                    new_orient = -new_orient

                new_translationalVelocity = rotate_by_quaternion(curr_transvel, iquaternion)
                
                if env.state_est:
                    state[1:5] = torch.FloatTensor(new_orient)
                    state[14:17] = torch.FloatTensor(new_translationalVelocity)
                    # state[0] = 1      # For use with StateEst. Replicate hack that height is always set to one on hardware.
                else:
                    state[2:6] = torch.FloatTensor(new_orient)
                    state[20:23] = torch.FloatTensor(new_translationalVelocity)

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

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    import sys, argparse, time
    parser = argparse.ArgumentParser()

    print_logo(subtitle="Maintained by Oregon State University's Dynamic Robotics Lab")

    """
        General arguments for configuring the environment
    """
    parser.add_argument("--traj", default="walking", type=str, help="reference trajectory to use. options are 'aslip', 'walking', 'stepping'")
    parser.add_argument("--clock_based", default=False, action='store_true')
    parser.add_argument("--state_est", default=True, action='store_true')
    parser.add_argument("--dyn_random", default=False, action='store_true')
    parser.add_argument("--no_delta", default=False, action='store_true')
    parser.add_argument("--reward", default="iros_paper", )
    parser.add_argument("--mirror", default=False, action='store_true')             # mirror actions or not

    """
        General arguments for configuring the logger
    """
    parser.add_argument("--run_name", default=None)                                    # run name


    if len(sys.argv) < 2:
        print("Usage: python apex.py [option]", sys.argv)

    elif sys.argv[1] == 'ars':
        """
            Utility for running Augmented Random Search.

        """
        from rl.algos.ars import run_experiment
        sys.argv.remove(sys.argv[1])
        parser.add_argument("--workers", type=int,    default=4)
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
        parser.add_argument("--logdir",               default="./trained_models/ars/", type=str)
        parser.add_argument("--seed",     "-s",       default=0, type=int)
        parser.add_argument("--env_name", "-e",       default="Hopper-v3")
        parser.add_argument("--average_every",        default=10, type=int)
        parser.add_argument("--save_model",   "-m",   default=None, type=str)               # where to save the trained model to
        parser.add_argument("--redis",                default=None)
        args = parser.parse_args()

        run_experiment(args)

    elif sys.argv[1] == 'ddpg' or sys.argv[1] == 'rdpg':

        recurrent = False if sys.argv[1] == 'ddpg' else True

        sys.argv.remove(sys.argv[1])
        """
            Utility for running Recurrent/Deep Deterministic Policy Gradient.
        """
        from rl.algos.dpg import run_experiment

        # Algo args
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
        parser.add_argument("--previous", type=str, default=None)

        # Logger args
        if recurrent:
            parser.add_argument("--logdir",                 default="./trained_models/rdpg/", type=str)
        else:
            parser.add_argument("--logdir",                 default="./trained_models/ddpg/", type=str)
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
        parser.add_argument("--logdir",       default="./trained_models/syncTD3/", type=str)
        parser.add_argument("--previous", type=str, default=None)                           # path to directory of previous policies for resuming training
        parser.add_argument("--env_name", default="Cassie-v0")                    # environment name
        parser.add_argument("--history", default=0, type=int)                                     # number of previous states to use as input
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

    elif sys.argv[1] == 'td3_async':
            
        sys.argv.remove(sys.argv[1])
        """
            Utility for running Twin-Delayed Deep Deterministic policy gradients (asynchronous).

        """
        from rl.algos.async_td3 import run_experiment

        # args common for actors and learners
        parser.add_argument("--env_name", default="Cassie-v0")                    # environment name
        parser.add_argument("--hidden_size", default=256)                         # neurons in hidden layer
        parser.add_argument("--history", default=0, type=int)                     # number of previous states to use as input

        # learner specific args
        parser.add_argument("--replay_size", default=1e8, type=int)               # Max size of replay buffer
        parser.add_argument("--max_timesteps", default=1e8, type=float)           # Max time steps to run environment for 1e8 == 100,000,000
        parser.add_argument("--batch_size", default=64, type=int)                 # Batch size for both actor and critic
        parser.add_argument("--discount", default=0.99, type=float)               # exploration/exploitation discount factor
        parser.add_argument("--tau", default=0.005, type=float)                   # target update rate (tau)
        parser.add_argument("--update_freq", default=2, type=int)                 # how often to update learner
        parser.add_argument("--evaluate_freq", default=5000, type=int)            # how often to evaluate learner
        parser.add_argument("--a_lr", type=float, default=3e-4)                   # Actor: Adam learning rate
        parser.add_argument("--c_lr", type=float, default=1e-4)                   # Critic: Adam learning rate

        # actor specific args
        parser.add_argument("--num_procs", default=30, type=int)                  # Number of actors
        parser.add_argument("--max_traj_len", type=int, default=400)              # max steps in each episode
        parser.add_argument("--start_timesteps", default=1e4, type=int)           # How many time steps purely random policy is run for
        parser.add_argument("--initial_load_freq", default=10, type=int)          # initial amount of time between loading global model
        parser.add_argument("--act_noise", default=0.3, type=float)               # Std of Gaussian exploration noise (used to be 0.1)
        parser.add_argument('--param_noise', type=bool, default=False)            # param noise
        parser.add_argument('--noise_scale', type=float, default=0.3)             # noise scale for param noise
        parser.add_argument("--taper_load_freq", type=bool, default=True)         # taper the load frequency over the course of training or not
        parser.add_argument("--viz_actors", default=False, action='store_true')   # Visualize actors in visdom or not

        # evaluator args
        parser.add_argument("--num_trials", default=10, type=int)                 # Number of evaluators
        parser.add_argument("--num_evaluators", default=10, type=int)             # Number of evaluators
        parser.add_argument("--viz_port", default=8097)                           # visdom server port
        parser.add_argument("--render_policy", type=bool, default=False)          # render during eval

        # misc args
        parser.add_argument("--policy_name", type=str, default="model")                 # name to save policy to
        parser.add_argument("--seed", type=int, default=1, help="RNG seed")
        parser.add_argument("--logger_name", type=str, default="tensorboard")           # logger to use (tensorboard or visdom)
        parser.add_argument("--logdir", type=str, default="./trained_models/td3_async/", help="Where to log diagnostics to")
        parser.add_argument("--previous", type=str, default=None)                           # path to directory of previous policies for resuming training
        parser.add_argument("--redis_address", type=str, default=None)                  # address of redis server (for cluster setups)

        args = parser.parse_args()

        run_experiment(args)

    elif sys.argv[1] == 'ppo':

        sys.argv.remove(sys.argv[1])
        """
            Utility for running Proximal Policy Optimization.

        """
        from rl.algos.ppo import run_experiment

        # general args
        parser.add_argument("--algo_name", default="ppo")                                   # algo name
        parser.add_argument("--env_name", "-e",   default="Cassie-v0")
        parser.add_argument("--logdir", type=str, default="./trained_models/ppo/")          # Where to log diagnostics to
        parser.add_argument("--previous", type=str, default=None)                           # path to directory of previous policies for resuming training
        parser.add_argument("--seed", default=0, type=int)                                  # Sets Gym, PyTorch and Numpy seeds
        parser.add_argument("--history", default=0, type=int)                                         # number of previous states to use as input
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
        parser.add_argument("--recurrent",   action='store_true')

        args = parser.parse_args()
        args.num_steps = args.num_steps // args.num_procs
        run_experiment(args)

    elif sys.argv[1] == 'eval':

        sys.argv.remove(sys.argv[1])

        parser.add_argument("--path", type=str, default="./trained_models/ppo/Cassie-v0/7b7e24-seed0/", help="path to folder containing policy and run details")
        parser.add_argument("--env_name", default="Cassie-v0", type=str)
        parser.add_argument("--traj_len", default=400, type=str)
        parser.add_argument("--history", default=0, type=int)                                         # number of previous states to use as input
        args = parser.parse_args()


        run_args = pickle.load(open(args.path + "experiment.pkl", "rb"))

        policy = torch.load(args.path + "actor.pt")

        eval_policy(policy, args, run_args)
        