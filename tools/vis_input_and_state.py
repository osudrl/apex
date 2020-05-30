import os, sys, argparse
sys.path.append("..") 

from cassie import CassieEnv, CassiePlayground
from rl.policies.actor import GaussianMLP_Actor

import matplotlib.pyplot as plt

import pickle
import numpy as np
import torch
import time

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def eval_policy(policy, args, run_args):

    aslip = True if run_args.traj == "aslip" else False

    cassie_env = CassieEnv(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, history=run_args.history, reward=run_args.reward)
    cassie_env.debug = args.debug
    visualize = not args.no_viz
    traj_len = args.traj_len

    if aslip:
        traj_info = [] # 
        traj_cmd_info = [] # what actually gets sent to robot as state
    robot_state_info = [] # robot's estimated state
    actual_state_info = [] # actual mujoco state of the robot

    state = torch.Tensor(cassie_env.reset_for_test())
    cassie_env.update_speed(2.0)
    print(cassie_env.speed)
    count, passed, done = 0, 1, False
    while count < traj_len and not done:

        if visualize:
            cassie_env.render()

        # Get action and act
        action = policy(state, True)
        action = action.data.numpy()
        state, reward, done, _ = cassie_env.step(action)
        state = torch.Tensor(state)

        print(reward)

        # print(cassie_env.phase)

        # See if end state reached
        if done or cassie_env.sim.qpos()[2] < 0.4:
            print(done)
            passed = 0
            print("failed")

        # Get trajectory info and robot info
        if aslip:
            a, b, c, d = cassie_env.get_traj_and_state_info()
            traj_info.append(a)
            traj_cmd_info.append(b)
        else:
            c, d = cassie_env.get_state_info()
        robot_state_info.append(c)
        actual_state_info.append(d)

        count += 1

    robot_state_info = robot_state_info[:-1]
    actual_state_info = actual_state_info[:-1]

    if aslip:

        traj_info = traj_info[:-1]
        traj_cmd_info = traj_cmd_info[:-1]

        traj_info = np.array(traj_info)
        traj_cmd_info = np.array(traj_cmd_info)
        robot_state_info = np.array(robot_state_info)
        actual_state_info = np.array(actual_state_info)

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        # print(traj_info)

        print(traj_info.shape)
        axs[0][0].set_title("XZ plane of traj_info")
        axs[0][0].plot(traj_info[:,0,0], traj_info[:,0,2], 'o-', label='cpos')
        axs[0][0].plot(traj_info[:,1,0], traj_info[:,1,2], 'o-', label='lpos')
        axs[0][0].plot(traj_info[:,2,0], traj_info[:,2,2], 'o-', label='rpos')

        print(traj_cmd_info.shape)
        axs[0][1].set_title("XZ plane of traj_cmd_info")
        axs[0][1].plot(traj_cmd_info[:,0,0], traj_cmd_info[:,0,2], label='cpos')
        axs[0][1].plot(traj_cmd_info[:,1,0], traj_cmd_info[:,1,2], label='lpos')
        axs[0][1].plot(traj_cmd_info[:,2,0], traj_cmd_info[:,2,2], label='rpos')

        print(robot_state_info.shape)
        axs[1][0].set_title("XZ plane of robot_state_info")
        axs[1][0].plot(robot_state_info[:,0,0], robot_state_info[:,0,2], label='cpos')
        axs[1][0].plot(robot_state_info[:,1,0], robot_state_info[:,1,2], label='lpos')
        axs[1][0].plot(robot_state_info[:,2,0], robot_state_info[:,2,2], label='rpos')

        print(actual_state_info.shape)
        axs[1][1].set_title("XZ plane of actual_state_info")
        axs[1][1].plot(actual_state_info[:,0,0], actual_state_info[:,0,2], label='cpos')
        axs[1][1].plot(actual_state_info[:,1,0], actual_state_info[:,1,2], label='lpos')
        axs[1][1].plot(actual_state_info[:,2,0], actual_state_info[:,2,2], label='rpos')

        plt.legend()
        plt.tight_layout()
        plt.show()

    else:

        robot_state_info = np.array(robot_state_info)
        actual_state_info = np.array(actual_state_info)

        fig, axs = plt.subplots(1, 2, figsize=(10, 10))

        print(robot_state_info.shape)
        axs[0].set_title("XZ plane of robot_state_info")
        axs[0].plot(robot_state_info[:,0,0], robot_state_info[:,0,2], label='cpos')
        axs[0].plot(robot_state_info[:,1,0], robot_state_info[:,1,2], label='lpos')
        axs[0].plot(robot_state_info[:,2,0], robot_state_info[:,2,2], label='rpos')

        print(actual_state_info.shape)
        axs[1].set_title("XZ plane of actual_state_info")
        axs[1].plot(actual_state_info[:,0,0], actual_state_info[:,0,2], label='cpos')
        axs[1].plot(actual_state_info[:,1,0], actual_state_info[:,1,2], label='lpos')
        axs[1].plot(actual_state_info[:,2,0], actual_state_info[:,2,2], label='rpos')

        plt.legend()
        plt.tight_layout()
        plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="../trained_models/ppo/Cassie-v0/IK_traj-aslip_aslip_old_2048_12288_seed-10/", help="path to folder containing policy and run details")
parser.add_argument("--traj_len", default=30, type=str)
parser.add_argument("--debug", default=False, action='store_true')
parser.add_argument("--no_viz", default=False, action='store_true')
parser.add_argument("--eval", default=True, action="store_false", help="Whether to call policy.eval() or not")

args = parser.parse_args()

run_args = pickle.load(open(args.path + "experiment.pkl", "rb"))

policy = torch.load(args.path + "actor.pt")

if args.eval:
    policy.eval()  # NOTE: for some reason the saved nodelta_neutral_stateest_symmetry policy needs this but it breaks all new policies...

eval_policy(policy, args, run_args)