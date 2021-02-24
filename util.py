import torch
import hashlib, os, pickle
from collections import OrderedDict
import sys, time
from cassie.quaternion_function import *

from tkinter import *
import multiprocessing as mp
from cassie.phase_function import LivePlot
import matplotlib.pyplot as plt

import tty
import termios
import select
import numpy as np
from cassie import CassieEnv, CassieMinEnv, CassiePlayground, CassieStandingEnv, CassieEnv_noaccel_footdist_omniscient, CassieEnv_footdist, CassieEnv_noaccel_footdist, CassieEnv_novel_footdist, CassieEnv_mininput
from cassie.cassiemujoco import CassieSim

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

def env_factory(path, traj="walking", simrate=50, phase_based=False, clock_based=True, state_est=True, dynamics_randomization=True, mirror=False, no_delta=False, ik_baseline=False, learn_gains=False, reward=None, history=0, fixed_speed=None, **kwargs):
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
    if path in ['Cassie-v0', 'CassieMin-v0', 'CassiePlayground-v0', 'CassieStandingEnv-v0', 'CassieNoaccelFootDistOmniscient', 'CassieFootDist', 'CassieNoaccelFootDist', 'CassieNoaccelFootDistNojoint', 'CassieNovelFootDist', 'CassieMinInput', 'CassieMinInputVelSidestep', 'CassieTurn', 'CassieTurn_no_orientadd', 'CassieClean']:
        from cassie import CassieEnv, CassieMinEnv, CassiePlayground, CassieStandingEnv, CassieEnv_noaccel_footdist_omniscient, CassieEnv_footdist, CassieEnv_noaccel_footdist, CassieEnv_noaccel_footdist_nojoint, CassieEnv_novel_footdist, CassieEnv_mininput, CassieEnv_mininput_vel_sidestep, CassieEnv_turn, CassieEnv_turn_no_orientadd, CassieEnv_clean

        if path == 'Cassie-v0':
            # env_fn = partial(CassieEnv, traj=traj, clock_based=clock_based, state_est=state_est, dynamics_randomization=dynamics_randomization, no_delta=no_delta, reward=reward, history=history)
            env_fn = partial(CassieEnv, traj=traj, simrate=simrate, phase_based=phase_based, clock_based=clock_based, state_est=state_est, dynamics_randomization=dynamics_randomization, no_delta=no_delta, learn_gains=learn_gains, ik_baseline=ik_baseline, reward=reward, history=history, fixed_speed=fixed_speed)
        elif path == 'CassieMin-v0':
            env_fn = partial(CassieMinEnv, traj=traj, simrate=simrate, phase_based=phase_based, clock_based=clock_based, state_est=state_est, dynamics_randomization=dynamics_randomization, no_delta=no_delta, learn_gains=learn_gains, ik_baseline=ik_baseline, reward=reward, history=history, fixed_speed=fixed_speed)
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
        elif path == "CassieNoaccelFootDistNojoint":
            env_fn = partial(CassieEnv_noaccel_footdist_nojoint, traj=traj, simrate=simrate, clock_based=clock_based, state_est=state_est, dynamics_randomization=dynamics_randomization, no_delta=no_delta, reward=reward, history=history)
        elif path == "CassieNovelFootDist":
            env_fn = partial(CassieEnv_novel_footdist, traj=traj, simrate=simrate, clock_based=clock_based, state_est=state_est, dynamics_randomization=dynamics_randomization, no_delta=no_delta, reward=reward, history=history)
        elif path == "CassieMinInput":
            env_fn = partial(CassieEnv_mininput, traj=traj, simrate=simrate, clock_based=clock_based, state_est=state_est, dynamics_randomization=dynamics_randomization, no_delta=no_delta, learn_gains=learn_gains, reward=reward, history=history)
        elif path == "CassieMinInputVelSidestep":
            env_fn = partial(CassieEnv_mininput_vel_sidestep, traj=traj, simrate=simrate, clock_based=clock_based, state_est=state_est, dynamics_randomization=dynamics_randomization, no_delta=no_delta, learn_gains=learn_gains, reward=reward, history=history)
        elif path == "CassieTurn":
            env_fn = partial(CassieEnv_turn, traj=traj, simrate=simrate, clock_based=clock_based, state_est=state_est, dynamics_randomization=dynamics_randomization, no_delta=no_delta, reward=reward, history=history)
        elif path == "CassieTurn_no_orientadd":
            env_fn = partial(CassieEnv_turn_no_orientadd, traj=traj, simrate=simrate, clock_based=clock_based, state_est=state_est, dynamics_randomization=dynamics_randomization, no_delta=no_delta, reward=reward, history=history)
        elif path == "CassieClean":
            env_fn = partial(CassieEnv_clean, simrate=simrate, dynamics_randomization=dynamics_randomization, reward=reward, history=history)
        else:
            print("Error: Unknown cassie environment")
            exit()
        # TODO for Yesh: make mirrored_obs an attribute of environment, configured based on setup parameters
        if mirror:
            from rl.envs.wrappers import SymmetricEnv
            mirror_act = [-5, -6, 7, 8, 9, -0.1, -1, 2, 3, 4]
            if learn_gains:
                mirror_act = [-5, -6, 7, 8, 9, -0.1, -1, 2, 3, 4, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 25, 26, 27, 28, 29, 20, 21, 22, 23, 24]
            env_fn = partial(SymmetricEnv, env_fn, mirrored_obs=env_fn().mirrored_obs, mirrored_act=mirror_act)

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
            if args.exchange_reward is not None:
                output_dir = args.previous[0:-1] + "_NEW-" + args.reward
            else:
                print(args.previous[0:-1])
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

# #TODO: Add pausing, and window quiting along with other render functionality
# def eval_policy(policy, args, run_args):

#     import tty
#     import termios
#     import select
#     import numpy as np
#     from cassie import CassieEnv, CassiePlayground, CassieStandingEnv, CassieEnv_noaccel_footdist_omniscient, CassieEnv_footdist, CassieEnv_noaccel_footdist
#     from cassie.cassiemujoco import CassieSim

#     def isData():
#         return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

#     max_traj_len = args.traj_len
#     visualize = not args.no_viz
#     print("env name: ", run_args.env_name)
#     if run_args.env_name is None:
#         env_name = args.env_name
#     else:
#         env_name = run_args.env_name
#     if env_name == "Cassie-v0":
#         env = CassieEnv(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, phase_based=run_args.phase_based, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
#     elif env_name == "CassiePlayground-v0":
#         env = CassiePlayground(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history, mission=args.mission)
#     elif env_name == "CassieNoaccelFootDistOmniscient":
#         env = CassieEnv_noaccel_footdist_omniscient(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
#     elif env_name == "CassieFootDist":
#         env = CassieEnv_footdist(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
#     elif env_name == "CassieNoaccelFootDist":
#         env = CassieEnv_noaccel_footdist(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
#     else:
#         env = CassieStandingEnv(state_est=run_args.state_est)
    
#     if args.debug:
#         env.debug = True

#     if args.terrain is not None and ".npy" in args.terrain:
#         env.sim = CassieSim("cassie_hfield.xml")
#         hfield_data = np.load(os.path.join("./cassie/cassiemujoco/terrains/", args.terrain))
#         env.sim.set_hfield_data(hfield_data.flatten())

#     print(env.reward_func)

#     if hasattr(policy, 'init_hidden_state'):
#         policy.init_hidden_state()

#     old_settings = termios.tcgetattr(sys.stdin)

#     orient_add = 0

#     slowmo = False

#     if visualize:
#         env.render()
#     render_state = True
#     try:
#         tty.setcbreak(sys.stdin.fileno())

#         state = env.reset_for_test()
#         done = False
#         timesteps = 0
#         eval_reward = 0
#         speed = 0.0

#         env.update_speed(speed)

#         while render_state:
        
#             if isData():
#                 c = sys.stdin.read(1)
#                 if c == 'w':
#                     speed += 0.1
#                 elif c == 's':
#                     speed -= 0.1
#                 elif c == 'j':
#                     env.phase_add += .1
#                     print("Increasing frequency to: {:.1f}".format(env.phase_add))
#                 elif c == 'h':
#                     env.phase_add -= .1
#                     print("Decreasing frequency to: {:.1f}".format(env.phase_add))
#                 elif c == 'l':
#                     orient_add += .1
#                     print("Increasing orient_add to: ", orient_add)
#                 elif c == 'k':
#                     orient_add -= .1
#                     print("Decreasing orient_add to: ", orient_add)
                
#                 elif c == 'x':
#                     env.swing_duration += .01
#                     print("Increasing swing duration to: ", env.swing_duration)
#                 elif c == 'z':
#                     env.swing_duration -= .01
#                     print("Decreasing swing duration  to: ", env.swing_duration)
#                 elif c == 'v':
#                     env.stance_duration += .01
#                     print("Increasing stance duration to: ", env.stance_duration)
#                 elif c == 'c':
#                     env.stance_duration -= .01
#                     print("Decreasing stance duration  to: ", env.stance_duration)

#                 elif c == '1':
#                     env.stance_mode = "zero"
#                     print("Stance mode: ", env.stance_mode)
#                 elif c == '2':
#                     env.stance_mode = "grounded"
#                     print("Stance mode: ", env.stance_mode)
#                 elif c == '3':
#                     env.stance_mode = "aerial"
#                     print("Stance mode: ", env.stance_mode)
#                 elif c == 'r':
#                     state = env.reset()
#                     speed = env.speed
#                     print("Resetting environment via env.reset()")
#                 elif c == 'p':
#                     push = 100
#                     push_dir = 2
#                     force_arr = np.zeros(6)
#                     force_arr[push_dir] = push
#                     env.sim.apply_force(force_arr)
#                 elif c == 't':
#                     slowmo = not slowmo
#                     print("Slowmo : ", slowmo)

#                 env.update_speed(speed)
#                 # print(speed)
#                 print("speed: ", env.speed)
            
#             if hasattr(env, 'simrate'):
#                 start = time.time()

#             if (not env.vis.ispaused()):
#                 # Update Orientation
#                 env.orient_add = orient_add
#                 # quaternion = euler2quat(z=orient_add, y=0, x=0)
#                 # iquaternion = inverse_quaternion(quaternion)

#                 # # TODO: Should probably not assume these indices. Should make them not hard coded
#                 # if env.state_est:
#                 #     curr_orient = state[1:5]
#                 #     curr_transvel = state[15:18]
#                 # else:
#                 #     curr_orient = state[2:6]
#                 #     curr_transvel = state[20:23]
                
#                 # new_orient = quaternion_product(iquaternion, curr_orient)

#                 # if new_orient[0] < 0:
#                 #     new_orient = -new_orient

#                 # new_translationalVelocity = rotate_by_quaternion(curr_transvel, iquaternion)
                
#                 # if env.state_est:
#                 #     state[1:5] = torch.FloatTensor(new_orient)
#                 #     state[15:18] = torch.FloatTensor(new_translationalVelocity)
#                 #     # state[0] = 1      # For use with StateEst. Replicate hack that height is always set to one on hardware.
#                 # else:
#                 #     state[2:6] = torch.FloatTensor(new_orient)
#                 #     state[20:23] = torch.FloatTensor(new_translationalVelocity)          
                    
#                 action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()

#                 state, reward, done, _ = env.step(action)

#                 # if env.lfoot_vel[2] < -0.6:
#                 #     print("left foot z vel over 0.6: ", env.lfoot_vel[2])
#                 # if env.rfoot_vel[2] < -0.6:
#                 #     print("right foot z vel over 0.6: ", env.rfoot_vel[2])
                
#                 eval_reward += reward
#                 timesteps += 1
#                 qvel = env.sim.qvel()
#                 # print("actual speed: ", np.linalg.norm(qvel[0:2]))
#                 # print("commanded speed: ", env.speed)

#                 if args.no_viz:
#                     yaw = quaternion2euler(new_orient)[2]
#                     print("stp = {}  yaw = {:.2f}  spd = {}  ep_r = {:.2f}  stp_r = {:.2f}".format(timesteps, yaw, speed, eval_reward, reward))

#             if visualize:
#                 render_state = env.render()
#             if hasattr(env, 'simrate'):
#                 # assume 40hz
#                 end = time.time()
#                 delaytime = max(0, 1000 / 40000 - (end-start))
#                 if slowmo:
#                     while(time.time() - end < delaytime*10):
#                         env.render()
#                         time.sleep(delaytime)
#                 else:
#                     time.sleep(delaytime)

#         print("Eval reward: ", eval_reward)

#     finally:
#         termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

class EvalProcessClass():
    def __init__(self, args, run_args):

        if run_args.phase_based and not args.no_viz:

            self.plot_pipe, plotter_pipe = mp.Pipe()
            self.plotter = LivePlot()
            self.plot_process = mp.Process(target=self.plotter, args=(plotter_pipe,), daemon=True)
            self.plot_process.start()

    #TODO: Add pausing, and window quiting along with other render functionality
    def eval_policy(self, policy, args, run_args):
        from cassie import CassieEnv, CassieMinEnv, CassiePlayground, CassieStandingEnv, CassieEnv_noaccel_footdist_omniscient, CassieEnv_footdist, CassieEnv_noaccel_footdist, CassieEnv_noaccel_footdist_nojoint, CassieEnv_novel_footdist, CassieEnv_mininput, CassieEnv_mininput_vel_sidestep, CassieEnv_turn, CassieEnv_turn_no_orientadd, CassieEnv_clean


        def print_input_update(e):
            print(f"\n\nstance dur.: {e.stance_duration:.2f}\t swing dur.: {e.swing_duration:.2f}\t stance mode: {e.stance_mode}\n")

        if run_args.phase_based and not args.no_viz:
            send = self.plot_pipe.send

        def isData():
            return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
        
        if args.debug:
            args.stats = False

        if args.reward is None:
            args.reward = run_args.reward

        max_traj_len = args.traj_len
        visualize = not args.no_viz
        print("env name: ", run_args.env_name)
        if run_args.env_name is None:
            env_name = args.env_name
        else:
            env_name = run_args.env_name
        if env_name == "Cassie-v0":
            env = CassieEnv(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, phase_based=run_args.phase_based, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
        elif env_name == "CassieMin-v0":
            env = CassieMinEnv(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, phase_based=run_args.phase_based, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
        elif env_name == "CassiePlayground-v0":
            env = CassiePlayground(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history, mission=args.mission)
        elif env_name == "CassieNoaccelFootDistOmniscient":
            env = CassieEnv_noaccel_footdist_omniscient(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
        elif env_name == "CassieFootDist":
            env = CassieEnv_footdist(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
        elif env_name == "CassieNoaccelFootDist":
            env = CassieEnv_noaccel_footdist(traj=run_args.traj, simrate=run_args.simrate, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
        elif env_name == "CassieNoaccelFootDistNojoint":
            env = CassieEnv_noaccel_footdist_nojoint(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
        elif env_name == "CassieNovelFootDist":
            env = CassieEnv_novel_footdist(traj=run_args.traj, simrate=run_args.simrate, clock_based=run_args.clock_based, state_est=run_args.state_est, dynamics_randomization=run_args.dyn_random, no_delta=run_args.no_delta, reward=args.reward, history=run_args.history)
        elif env_name == "CassieMinInput":
            env = CassieEnv_mininput(traj=run_args.traj, simrate=run_args.simrate, clock_based=run_args.clock_based, state_est=run_args.state_est, dynamics_randomization=run_args.dyn_random, no_delta=run_args.no_delta, learn_gains=run_args.learn_gains, reward=args.reward, history=run_args.history)
        elif env_name == "CassieMinInputVelSidestep":
            env = CassieEnv_mininput_vel_sidestep(traj=run_args.traj, simrate=run_args.simrate, clock_based=run_args.clock_based, state_est=run_args.state_est, dynamics_randomization=run_args.dyn_random, no_delta=run_args.no_delta, reward=args.reward, history=run_args.history)
        elif env_name == "CassieTurn":
            env = CassieEnv_turn(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
        elif env_name == "CassieTurn_no_orientadd":
            env = CassieEnv_turn_no_orientadd(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
        elif env_name == "CassieClean":
            env = CassieEnv_clean(simrate=run_args.simrate, dynamics_randomization=run_args.dyn_random, reward=args.reward, history=run_args.history)
        else:
            env = CassieStandingEnv(state_est=run_args.state_est)
        
        if args.debug:
            env.debug = True

        if args.terrain is not None and ".npy" in args.terrain:
            env.sim = CassieSim("cassie_hfield.xml")
            hfield_data = np.load(os.path.join("./cassie/cassiemujoco/terrains/", args.terrain))
            env.sim.set_hfield_data(hfield_data.flatten())

        # print(env.reward_func)
        # env.reward_func = "speedmatchavg_footvarclock_footorient_stablepel_hiprollyawvel_smoothact_torquecost_reward"
        # print()

        if hasattr(policy, 'init_hidden_state'):
            policy.init_hidden_state()

        old_settings = termios.tcgetattr(sys.stdin)

        orient_add = 0

        slowmo = False

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
            # slowmotion = False
            slow_factor = 3
            curr_slow = 0

            env.update_speed(speed)

            while render_state:
            
                if isData():
                    c = sys.stdin.read(1)
                    if c == 'w':
                        speed += 0.1
                    elif c == 's':
                        speed -= 0.1
                    elif c == 'a':
                        env.side_speed -= 0.1
                    elif c == 'd':
                        env.side_speed += 0.1
                    elif c == 'j':
                        env.phase_add += .1
                        # print("Increasing frequency to: {:.1f}".format(env.phase_add))
                    elif c == 'h':
                        env.phase_add -= .1
                        # print("Decreasing frequency to: {:.1f}".format(env.phase_add))
                    elif c == 'l':
                        orient_add -= .1
                        # print("Increasing orient_add to: ", orient_add)
                    elif c == 'k':
                        orient_add += .1
                        # print("Decreasing orient_add to: ", orient_add)
                    
                    elif c == 'x':
                        env.swing_duration += .01
                        print_input_update(env)
                    elif c == 'z':
                        env.swing_duration -= .01
                        print_input_update(env)
                    elif c == 'v':
                        env.stance_duration += .01
                        print_input_update(env)
                    elif c == 'c':
                        env.stance_duration -= .01
                        print_input_update(env)

                    elif c == '1':
                        env.stance_mode = "zero"
                        print_input_update(env)
                    elif c == '2':
                        env.stance_mode = "grounded"
                        print_input_update(env)
                    elif c == '3':
                        env.stance_mode = "aerial"
                        
                    elif c == 'r':
                        state = env.reset_for_test()
                        speed = env.speed
                        if hasattr(policy, 'init_hidden_state'):
                            policy.init_hidden_state()
                        print("Resetting environment via env.reset()")
                    elif c == 'p':
                        push = 100
                        push_dir = 2
                        force_arr = np.zeros(6)
                        force_arr[push_dir] = push
                        env.sim.apply_force(force_arr)
                    elif c == 't':
                        slowmo = not slowmo
                        print("Slowmo : \n", slowmo)

                    env.update_speed(speed)
                    # print(speed)

                    if env.phase_based and visualize:
                        send((env.swing_duration, env.stance_duration, env.strict_relaxer, env.stance_mode, env.have_incentive))
                
                if args.stats:
                    print(f"act spd: {env.sim.qvel()[0]:.2f}   cmd speed: {env.speed:.2f}   phase add: {env.phase_add:.2f}   orient add: {orient_add:.2f}", end="\r")
                    # print(f"act spd: {env.sim.qvel()[0]:.2f}\t cmd speed: {env.speed:.2f}\t phase add: {env.phase_add:.2f}\t orient add: {orient_add}", end="\r")

                if hasattr(env, 'simrate'):
                    start = time.time()

                if (not env.vis.ispaused()) and (not slowmo or (slowmo and curr_slow == slow_factor)):
                    curr_slow = 0
                    # Update Orientation
                    # env.orient_add = orient_add
                    env.turn_rate = orient_add / 10
                    # print(env.orient_add)
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
                    foot_pos = np.zeros(6)
                    env.sim.foot_pos(foot_pos)
                    # print(foot_pos[2], foot_pos[5])

                    eval_reward += reward
                    timesteps += 1
                    qvel = env.sim.qvel()
                    # print("actual speed: ", np.linalg.norm(qvel[0:2]))
                    # print("commanded speed: ", env.speed)

                    if args.no_viz:
                        yaw = quaternion2euler(new_orient)[2]
                        print("stp = {}  yaw = {:.2f}  spd = {}  ep_r = {:.2f}  stp_r = {:.2f}".format(timesteps, yaw, speed, eval_reward, reward))

                curr_slow += 1
                if visualize:
                    render_state = env.render()
                if hasattr(env, 'simrate'):
                    # assume 40hz
                    end = time.time()
                    delaytime = max(0, 1000 / 40000 - (end-start))
                    # if slowmo:
                    #     while(time.time() - end < delaytime*10):
                    #         env.render()
                    #         time.sleep(delaytime)
                    # else:
                    time.sleep(delaytime)

            print("Eval reward: ", eval_reward)

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def collect_data(policy, args, run_args):
    wait_steps = 0
    num_cycles = 35
    speed = 0.0

    if args.reward is None:
        args.reward = run_args.reward

    if run_args.env_name is None:
        env_name = args.env_name
    else:
        env_name = run_args.env_name
    if env_name == "Cassie-v0":
        env = CassieEnv(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, phase_based=run_args.phase_based, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
    elif env_name == "CassiePlayground-v0":
        env = CassiePlayground(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history, mission=args.mission)
    elif env_name == "CassieNoaccelFootDistOmniscient":
        env = CassieEnv_noaccel_footdist_omniscient(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
    elif env_name == "CassieFootDist":
        env = CassieEnv_footdist(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
    elif env_name == "CassieNoaccelFootDist":
        env = CassieEnv_noaccel_footdist(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
    elif env_name == "CassieNoaccelFootDistNojoint":
        env = CassieEnv_noaccel_footdist_nojoint(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
    else:
        env = CassieStandingEnv(state_est=run_args.state_est)

    if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()

    state = torch.Tensor(env.reset_for_test())
    env.update_speed(speed)
    env.render()

    time_log, speed_log, grf_log = [], [], []

    # print("iros: ", iros_env.simrate, iros_env.phaselen)
    # print("aslip: ", aslip_env.simrate, aslip_env.phaselen)

    with torch.no_grad():

        # Run few cycles to stabilize (do separate incase two envs have diff phaselens)
        for i in range(wait_steps):
            action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
            state, reward, done, _ = env.step(action)
            env.render()
            print(f"act spd: {env.sim.qvel()[0]:.2f}\t cmd speed: {env.speed:.2f}")
            # curr_qpos = aslip_env.sim.qpos()
            # print("curr height: ", curr_qpos[2])

        # Collect actual data (avg foot force over simrate)
        # start_time = time.time()
        # for i in range(math.floor(num_cycles*env.phaselen)+1):
        #     action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
        #     state, reward, done, _ = env.step(action)
        #     speed_log.append([env.speed, env.sim.qvel()[0]])

        #     # grf_log.append(np.concatenate([[time.time()-start_time],env.sim.get_foot_forces()]))
        #     grf_log.append([time.time()-start_time, env.l_foot_frc, env.r_foot_frc])
        #     env.render()
        #     print(f"act spd: {env.sim.qvel()[0]:.2f}\t cmd speed: {env.speed:.2f}")
        #     # curr_qpos = aslip_env.sim.qpos()
        #     # print("curr height: ", curr_qpos[2])

        # Collect actual data (foot force each sim step)
        print("Start actual data")
        start_time = time.time()
        for i in range(num_cycles):
            for j in range(math.floor(env.phaselen)+1):
                action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
                for k in range(env.simrate):
                    env.step_simulation(action)
                    time_log.append(time.time()-start_time)
                    speed_log.append([env.speed, env.sim.qvel()[0]])
                    grf_log.append(env.sim.get_foot_forces())
                
                env.time += 1
                env.phase += env.phase_add
                if env.phase > env.phaselen:
                    env.phase = 0
                    env.counter += 1
                state = env.get_full_state()
                env.speed = i * 0.1
                env.render()
        
        time_log, speed_log, grf_log = map(np.array, (time_log, speed_log, grf_log))
        print(speed_log.shape)
        print(grf_log.shape)

    ### Process the data

    # average data and get std dev
    mean_speed = np.mean(speed_log, axis=0)
    stddev_speed = np.mean(speed_log, axis=0)
    mean_grfs = np.mean(grf_log[:, 1:], axis=0)
    stddev_grfs = np.std(grf_log[:, 1:], axis=0)
    print(mean_speed)
    print(stddev_speed)
    print(mean_grfs)
    print(stddev_grfs)

    ### Save the data
    output = {
        "time": time_log,
        "speed": speed_log,
        "grfs": grf_log
    }
    with open(os.path.join(args.path, "collect_data.pkl"), "wb") as f:
        pickle.dump(output, f)

    with open(os.path.join(args.path, "collect_data.pkl"), "rb") as f:
        data = pickle.load(f)
    
    time_data = data["time"]
    speed_data = data["speed"]
    grf_data = data["grfs"]

    ### Plot the data
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    axs[0].plot(time_data, speed_data, label="commanded speed")
    axs[0].plot(time_data, speed_data, label="actual speed")
    axs[0].set_title("Speed")
    axs[0].set_xlabel("Time (Seconds)")
    axs[0].set_ylabel("m/s")
    axs[1].plot(time_data, grf_log[:,0], label="sim left foot")
    axs[1].plot(time_data, grf_log[:,1], label="sim right foot")
    axs[1].set_title("GRFs")
    axs[1].set_xlabel("Time (Seconds)")
    axs[1].set_ylabel("Newtons")
    plt.legend(loc="upper right")
    plt.show()

# Rule for curriculum learning is that env observation space should be the same (so attributes like env.clock_based or env.state_est shouldn't be different and are forced to be same here)
# deal with loading hyperparameters of previous run continuation
def parse_previous(args):
    if args.previous is not None:
        run_args = pickle.load(open(args.previous + "experiment.pkl", "rb"))
        args.env_name = run_args.env_name
        args.traj = run_args.traj
        args.phase_based = run_args.phase_based
        args.clock_based = run_args.clock_based
        args.state_est = run_args.state_est
        args.no_delta = run_args.no_delta
        args.recurrent = run_args.recurrent
        args.learn_gains = run_args.learn_gains
        args.ik_baseline = run_args.ik_baseline,
        if args.env_name == "CassiePlayground-v0":
            args.reward = "command"
            args.run_name = run_args.run_name + "-playground"
        if args.exchange_reward != None:
            args.reward = args.exchange_reward
            args.run_name = run_args.run_name + "_NEW-" + args.reward
        else:
            args.reward = run_args.reward
            args.run_name = run_args.run_name + "--cont"
    return args
