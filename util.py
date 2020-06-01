import torch
import hashlib, os, pickle
from collections import OrderedDict
import sys, time
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

def env_factory(path, traj="walking", simrate=50, clock_based=True, state_est=True, dynamics_randomization=True, mirror=False, no_delta=False, reward=None, history=0, **kwargs):
    from functools import partial

    """
    Returns an *uninstantiated* environment constructor.

    Since environments containing cpointers (e.g. Mujoco envs) can't be serialized, 
    this allows us to pass their constructors to Ray remote functions instead 
    (since the gym registry isn't shared across ray subprocesses we can't simply 
    pass gym.make() either)

    Note: env.unwrapped.spec is never set, if that matters for some reason.
    """
    # if history > 0:
    #   raise NotImplementedError

    # Custom Cassie Environment
    if path in ['Cassie-v0', 'CassiePlayground-v0', 'CassieStandingEnv-v0', 'CassieNoaccelFootDistOmniscient', 'CassieFootDist', 'CassieNoaccelFootDist']:
        from cassie import CassieEnv, CassiePlayground, CassieStandingEnv, CassieEnv_noaccel_footdist_omniscient, CassieEnv_footdist, CassieEnv_noaccel_footdist

        if path == 'Cassie-v0':
            env_fn = partial(CassieEnv, traj=traj, simrate=simrate, clock_based=clock_based, state_est=state_est, dynamics_randomization=dynamics_randomization, no_delta=no_delta, reward=reward, history=history)
        elif path == 'CassiePlayground-v0':
            env_fn = partial(CassiePlayground, traj=traj, simrate=simrate, clock_based=clock_based, state_est=state_est, dynamics_randomization=dynamics_randomization, no_delta=no_delta, reward=reward, history=history)
        elif path == 'CassieStandingEnv-v0':
            env_fn = partial(CassieStandingEnv, simrate=simrate, state_est=state_est)
        elif path == 'CassieNoaccelFootDistOmniscient':
            env_fn = partial(CassieEnv_noaccel_footdist_omniscient, simrate=simrate, traj=traj, clock_based=clock_based, state_est=state_est, dynamics_randomization=True, no_delta=no_delta, reward=reward, history=history)
        elif path == 'CassieFootDist':
            env_fn = partial(CassieEnv_footdist, traj=traj, simrate=simrate, clock_based=clock_based, state_est=state_est, dynamics_randomization=dynamics_randomization, no_delta=no_delta, reward=reward, history=history)
        elif path == 'CassieNoaccelFootDist':
            env_fn = partial(CassieEnv_noaccel_footdist, traj=traj, simrate=simrate, clock_based=clock_based, state_est=state_est, dynamics_randomization=dynamics_randomization, no_delta=no_delta, reward=reward, history=history)


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
    env_name = str(arg_dict['env_name'])

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

#TODO: Add pausing, and window quiting along with other render functionality
def eval_policy(policy, args, run_args):

    import tty
    import termios
    import select
    import numpy as np
    from cassie import CassieEnv, CassiePlayground, CassieStandingEnv, CassieEnv_noaccel_footdist_omniscient, CassieEnv_footdist, CassieEnv_noaccel_footdist

    def isData():
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    max_traj_len = args.traj_len
    visualize = not args.no_viz
    print("env name: ", run_args.env_name)
    if run_args.env_name is None:
        env_name = args.env_name
    else:
        env_name = run_args.env_name
    if env_name == "Cassie-v0":
        env = CassieEnv(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
    elif env_name == "CassiePlayground-v0":
        env = CassiePlayground(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history, mission=args.mission)
    elif env_name == "CassieNoaccelFootDistOmniscient":
        env = CassieEnv_noaccel_footdist_omniscient(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
    elif env_name == "CassieFootDist":
        env = CassieEnv_footdist(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
    elif env_name == "CassieNoaccelFootDist":
        env = CassieEnv_noaccel_footdist(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
    else:
        env = CassieStandingEnv(state_est=run_args.state_est)
    
    if args.debug:
        env.debug = True

    print(env.reward_func)

    if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()

    old_settings = termios.tcgetattr(sys.stdin)

    orient_add = 0

    if visualize:
        env.render()
    render_state = True
    try:
        tty.setcbreak(sys.stdin.fileno())

        state = env.reset_for_test()
        done = False
        timesteps = 0
        eval_reward = 0
        speed = 0.0

        while render_state:
        
            if isData():
                c = sys.stdin.read(1)
                if c == 'w':
                    speed += 0.1
                elif c == 's':
                    speed -= 0.1
                elif c == 'j':
                    env.phase_add += .1
                    print("Increasing frequency to: {:.1f}".format(env.phase_add))
                elif c == 'h':
                    env.phase_add -= .1
                    print("Decreasing frequency to: {:.1f}".format(env.phase_add))
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

                env.update_speed(speed)
                # print(speed)
                print("speed: ", env.speed)
            
            if hasattr(env, 'simrate'):
                start = time.time()

            if (not env.vis.ispaused()):
                # Update Orientation
                env.orient_add = orient_add
                # quaternion = euler2quat(z=orient_add, y=0, x=0)
                # iquaternion = inverse_quaternion(quaternion)

                # # TODO: Should probably not assume these indices. Should make them not hard coded
                # if env.state_est:
                #     curr_orient = state[1:5]
                #     curr_transvel = state[15:18]
                # else:
                #     curr_orient = state[2:6]
                #     curr_transvel = state[20:23]
                
                # new_orient = quaternion_product(iquaternion, curr_orient)

                # if new_orient[0] < 0:
                #     new_orient = -new_orient

                # new_translationalVelocity = rotate_by_quaternion(curr_transvel, iquaternion)
                
                # if env.state_est:
                #     state[1:5] = torch.FloatTensor(new_orient)
                #     state[15:18] = torch.FloatTensor(new_translationalVelocity)
                #     # state[0] = 1      # For use with StateEst. Replicate hack that height is always set to one on hardware.
                # else:
                #     state[2:6] = torch.FloatTensor(new_orient)
                #     state[20:23] = torch.FloatTensor(new_translationalVelocity)          
                    
                action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
                state, reward, done, _ = env.step(action)

                # if env.lfoot_vel[2] < -0.6:
                #     print("left foot z vel over 0.6: ", env.lfoot_vel[2])
                # if env.rfoot_vel[2] < -0.6:
                #     print("right foot z vel over 0.6: ", env.rfoot_vel[2])
                
                eval_reward += reward
                timesteps += 1
                qvel = env.sim.qvel()
                # print("actual speed: ", np.linalg.norm(qvel[0:2]))
                # print("commanded speed: ", env.speed)

                if args.no_viz:
                    yaw = quaternion2euler(new_orient)[2]
                    print("stp = {}  yaw = {:.2f}  spd = {}  ep_r = {:.2f}  stp_r = {:.2f}".format(timesteps, yaw, speed, eval_reward, reward))

            if visualize:
                render_state = env.render()
            if hasattr(env, 'simrate'):
                # assume 30hz (hack)
                end = time.time()
                delaytime = max(0, 1000 / 30000 - (end-start))
                time.sleep(delaytime)

        print("Eval reward: ", eval_reward)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

# deal with loading hyperparameters of previous run continuation
def parse_previous(args):
    if args.previous is not None:
        run_args = pickle.load(open(args.previous + "experiment.pkl", "rb"))
        args.traj = run_args.traj
        args.clock_based = run_args.clock_based
        args.state_est = run_args.state_est
        args.dyn_random = run_args.dyn_random
        args.no_delta = run_args.no_delta
        args.mirror = run_args.mirror
        args.recurrent = run_args.recurrent
        if args.env_name == "CassiePlayground-v0":
            args.reward = "command"
            args.run_name = run_args.run_name + "-playground"
    
    return args