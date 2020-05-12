import os, sys, argparse
sys.path.append("..") 

from cassie import CassieEnv, CassiePlayground
from rl.policies.actor import GaussianMLP_Actor

import matplotlib.pyplot as plt

import pickle
import numpy as np
import torch
import time

def eval_policy(policy, args, run_args):

    cassie_env = CassieEnv(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, history=run_args.history, reward="aslip")
    cassie_env.debug = args.debug
    visualize = not args.no_viz
    traj_len = args.traj_len

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
        a, b, c, d = cassie_env.get_traj_and_state_info()
        traj_info.append(a)
        traj_cmd_info.append(b)
        robot_state_info.append(c)
        actual_state_info.append(d)

        count += 1

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



parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=None, help="path to folder containing policy and run details")
parser.add_argument("--traj_len", default=400, type=str)
parser.add_argument("--debug", default=False, action='store_true')
parser.add_argument("--no_viz", default=False, action='store_true')

args = parser.parse_args()

args.path = "../trained_models/ppo/Cassie-v0/IK_traj-aslip_aslip_comorientheight_1024_8192_seed-0_NEW-aslip_TaskSpaceMujoco/"
args.path = "../trained_models/ppo/Cassie-v0/IK_traj-aslip_aslip_comorientheight_1024_8192_seed-0_NEW-aslip_strict/"

run_args = pickle.load(open(args.path + "experiment.pkl", "rb"))

policy = torch.load(args.path + "actor.pt")
# policy.eval()  # NOTE: for some reason the saved nodelta_neutral_stateest_symmetry policy needs this but it breaks all new policies...

eval_policy(policy, args, run_args)



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


# trajectory_type = input("Enter the type of trajectory [iros_paper, aslip, mission]:\n")
"""

trajectory_type = "aslipRT"


if trajectory_type == "iros_paper":
    env = CassieEnv(traj="walking")
    traj = env.trajectory

    time = traj.time
    qpos = traj.qpos
    qvel = traj.qvel
    torque = traj.torque
    mpos = traj.mpos
    mvel = traj.mvel
    
    datas = []
    datas.append( (qpos, "qpos", list(map(str, range(35))) ))
    datas.append((qvel, "qvel", list(map(str, range(32))) ))
    datas.append((torque, "torque", ["l hip roll", "l hip yaw", "l hip pitch", "l knee", "l,foot", "r hip roll", "r hip yaw", "r hip pitch", "r knee", "r foot"]) )
    datas.append((mpos, "mpos", ["l hip roll", "l hip yaw", "l hip pitch", "l knee", "l,foot", "r hip roll", "r hip yaw", "r hip pitch", "r knee", "r foot"]) )
    datas.append((mvel, "mvel", ["l hip roll", "l hip yaw", "l hip pitch", "l knee", "l,foot", "r hip roll", "r hip yaw", "r hip pitch", "r knee", "r foot"]) )

    fig, axs = plt.subplots(5,1, sharex=True)
    for i, data in enumerate(datas):
        axs[i].plot(time, data[0], label=data[2])
        axs[i].set_ylabel(data[1])
        axs[i].set_title(data[1])
        # axs[i].legend()
    axs[-1].set_xlabel("time")
    plt.tight_layout()
    plt.show()

elif trajectory_type == "aslip":
    env = CassieEnv(traj="aslip")

    speed = float(input("Speed to visualize? [0.0, 0.1, 0.2, ... 1.9, 2.0]:\n"))
    assert(isinstance(speed, float) or isinstance(speed, int))
    env.update_speed(speed)
    print("speed = ", env.speed)

    traj = env.trajectory

    time = np.arange(0, traj.length*(1/(2000 / env.simrate)), step=(1/(2000 / env.simrate))) # time separated by 1/freq, where freq = 2000/simrate
    print(time)
    qpos = traj.qpos
    qvel = traj.qvel
    rpos = traj.rpos
    rvel = traj.rvel
    lpos = traj.lpos
    lvel = traj.lvel
    cpos = traj.cpos
    cvel = traj.cvel
    
    offset = 0.2

    cpos[:,2] += offset
    # need to update these because they 
    lpos[:,2] -= offset
    rpos[:,2] -= offset

    datas = []
    datas.append( (qpos, "qpos") )
    datas.append( (qvel, "qvel") )

    fig, axs = plt.subplots(2,1, sharex=True)

    # plots of qpos and qvel
    for i, data in enumerate(datas):
        axs[i].plot(time, data[0])
        axs[i].set_ylabel(data[1])
        axs[i].set_title(data[1])
    plt.tight_layout()
    plt.show()

    # 3d plot of path
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(cpos[:,0], cpos[:,1], cpos[:,2], 'o-', label='com')
    ax.plot(rpos[:,0]+cpos[:,0], rpos[:,1]+cpos[:,1], rpos[:,2]+cpos[:,2], 'o-', label='right')
    ax.plot(lpos[:,0]+cpos[:,0], lpos[:,1]+cpos[:,1], lpos[:,2]+cpos[:,2], 'o-', label='left')
    set_axes_equal(ax)
    ax.set_zlim3d([0.0, 1.0])

    # 3d plot of each foot
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(rpos[:,0], rpos[:,1], rpos[:,2], 'o-', label='right')
    ax.plot(lpos[:,0], lpos[:,1], lpos[:,2], 'o-', label='left')
    set_axes_equal(ax)

    plt.show()

elif trajectory_type == "aslipRT":
    policy = torch.load("../trained_models/ppo/Cassie-v0/traj-aslip_aslip_64_5096/" + "actor.pt")

    eval_policy(policy)

"""