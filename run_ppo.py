import torch
import hashlib, os, pickle
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

    logger = SummaryWriter(output_dir, flush_secs=60) # flush_secs=0.1 actually slows down quite a bit, even on parallelized set ups
    print("Logging to " + color.BOLD + color.ORANGE + str(output_dir) + color.END)

    logger.dir = output_dir
    return logger


if __name__ == "__main__":
    import sys, argparse, time
    parser = argparse.ArgumentParser()

    print_logo(subtitle="Maintained by Oregon State University's Dynamic Robotics Lab")

    """
        General arguments for configuring the environment
    """
    parser.add_argument("--traj", default="walking", type=str, help="reference trajectory to use. options are 'aslip', 'walking', 'stepping'")
    parser.add_argument("--clock_based", default=True, action='store_true')
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

