import torch
import sys, pickle, argparse
from util.logo import print_logo
from util.log import parse_previous
from util.eval import EvalProcessClass

if __name__ == "__main__":

    print_logo(subtitle="Maintained by Oregon State University's Dynamic Robotics Lab")
    parser = argparse.ArgumentParser()

    """
        General arguments for configuring the environment
    """
    # command input, state input, env attributes
    parser.add_argument("--command_profile", default="clock", type=str.lower, choices=["clock", "phase", "traj"])
    parser.add_argument("--input_profile", default="full", type=str.lower, choices=["full", "min"])
    parser.add_argument("--simrate", default=50, type=int, help="simrate of environment")
    parser.add_argument("--not_dyn_random", default=True, action='store_false', dest='dyn_random')
    parser.add_argument("--learn_gains", default=False, action='store_true', dest='learn_gains')             # learn PD gains or not
    # attributes for trajectory based environments
    parser.add_argument("--traj", default="walking", type=str, help="reference trajectory to use. options are 'aslip', 'walking', 'stepping'")
    parser.add_argument("--not_no_delta", default=True, action='store_false', dest='no_delta')
    parser.add_argument("--ik_baseline", default=False, action='store_true', dest='ik_baseline')             # use ik as baseline for aslip + delta policies?
    # mirror loss and reward
    parser.add_argument("--not_mirror", default=True, action='store_false', dest='mirror')             # mirror actions or not
    parser.add_argument("--reward", default=None, type=str)                                             # reward to use. this is a required argument

    """
        General arguments for configuring the logger
    """
    parser.add_argument("--env_name", default="Cassie-v0")                             # environment name
    parser.add_argument("--run_name", default=None)                                    # run name

    """
        General arguments for Curriculum Learning
    """
    parser.add_argument("--exchange_reward", default=None)                              # Can only be used with previous (below)
    parser.add_argument("--previous", type=str, default=None)                           # path to directory of previous policies for resuming training

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

        args = parser.parse_args()

        args.recurrent = recurrent

        args = parse_previous(args)

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
        args = parse_previous(args)

        run_experiment(args)

    elif sys.argv[1] == 'td3_async':
            
        sys.argv.remove(sys.argv[1])
        """
            Utility for running Twin-Delayed Deep Deterministic policy gradients (asynchronous).

        """
        from rl.algos.async_td3 import run_experiment

        # args common for actors and learners
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
        args = parse_previous(args)

        run_experiment(args)

    elif sys.argv[1] == 'ppo':

        sys.argv.remove(sys.argv[1])
        """
            Utility for running Proximal Policy Optimization.

        """
        from rl.algos.ppo import run_experiment

        # general args
        parser.add_argument("--logdir", type=str, default="./trained_models/ppo/")          # Where to log diagnostics to
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
        parser.add_argument("--anneal", default=1.0, action='store_true', help="anneal rate for stddev")
        parser.add_argument("--learn_stddev", default=False, action='store_true', help="learn std_dev or keep it fixed")
        parser.add_argument("--std_dev", type=int, default=-1.5, help="exponent of exploration std_dev")
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
        parser.add_argument("--bounded",   type=bool, default=False)
        args = parser.parse_args()

        args = parse_previous(args)

        run_experiment(args)

    elif sys.argv[1] == 'eval':

        sys.argv.remove(sys.argv[1])

        parser.add_argument("--path", type=str, default="./trained_models/nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2/", help="path to folder containing policy and run details")
        parser.add_argument("--traj_len", default=400, type=str)
        parser.add_argument("--history", default=0, type=int)                                    # number of previous states to use as input
        parser.add_argument("--mission", default="default", type=str)                            # only used by playground environment
        parser.add_argument("--terrain", default=None, type=str)                                 # hfield file name (terrain to use)
        parser.add_argument("--debug", default=False, action='store_true')
        parser.add_argument("--no_stats", dest="stats", default=True, action='store_false')
        parser.add_argument("--no_viz", default=False, action='store_true')

        args = parser.parse_args()

        run_args = pickle.load(open(args.path + "experiment.pkl", "rb"))

        policy = torch.load(args.path + "actor.pt")
        policy.eval()

        # eval_policy(policy, args, run_args)
        # eval_policy_input_viz(policy, args, run_args)
        ev = EvalProcessClass(args, run_args)
        ev.eval_policy(policy, args, run_args)
