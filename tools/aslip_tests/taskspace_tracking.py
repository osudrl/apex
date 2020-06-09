"""
Measures the taskspace tracking error for aslip policies individually for each speed the policy accepts
"""

import os, sys, argparse
sys.path.append("../..") 

from cassie import CassieEnv, CassiePlayground
from rl.policies.actor import GaussianMLP_Actor

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def eval_policy(policy, args, run_args, testing_speed):

    max_traj_len = args.traj_len + args.ramp_up
    visualize = args.viz
    # run_args.dyn_random = True

    env = CassieEnv(traj="aslip", state_est=run_args.state_est, no_delta=run_args.no_delta, learn_gains=run_args.learn_gains, ik_baseline=run_args.ik_baseline, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward="aslip_old", history=run_args.history)
    
    if args.debug:
        env.debug = True

    print(env.reward_func)

    if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()

    orient_add = 0

    if visualize:
        env.render()
    render_state = True

    state = env.reset_for_test()
    done = False
    timesteps = 0
    eval_reward = 0

    # Data to track
    time_log = []              # time in seconds
    traj_info = []         # Information from reference trajectory library
    actual_state_info = [] # actual mujoco state of the robot
    l_footstep = []      # (time, left foot desired placement, left foot actual placement)
    r_footstep = []      # (time, right foot desired placement, right foot actual placement)
    # footstep = []

    env.update_speed(testing_speed)
    print(env.speed)

    while timesteps < max_traj_len:
    
        if hasattr(env, 'simrate'):
            start = time.time()

        # if (not env.vis.ispaused()):
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

        # if timesteps > args.ramp_up:
        # print(env.counter)
        a, _, _, d = env.get_traj_and_state_info()
        traj_info.append(a)
        actual_state_info.append(d)
        time_log.append(timesteps / 40)

        if a[1][2] == 0.0:
            l_footstep.append(np.linalg.norm(a[1] - d[1]))
        elif a[2][2] == 0.0:
            r_footstep.append(np.linalg.norm(a[2] - d[2]))

        # if traj_info[]

        # if env.lfoot_vel[2] < -0.6:
        #     print("left foot z vel over 0.6: ", env.lfoot_vel[2])
        # if env.rfoot_vel[2] < -0.6:
        #     print("right foot z vel over 0.6: ", env.rfoot_vel[2])
        
        eval_reward += reward
        timesteps += 1
        qvel = env.sim.qvel()
        # print("actual speed: ", np.linalg.norm(qvel[0:2]))
        # print("commanded speed: ", env.speed)

        if visualize:
            render_state = env.render()
        # if hasattr(env, 'simrate'):
        #     # assume 40hz
        #     end = time.time()
        #     delaytime = max(0, 1000 / 40000 - (end-start))
        #     time.sleep(delaytime)

    actual_state_info = actual_state_info[:-1]
    traj_info = traj_info[:-1]
    time_log = time_log[:-1]

    print("Eval reward: ", eval_reward)

    traj_info = np.array(traj_info)
    actual_state_info = np.array(actual_state_info)
    time_log = np.array(time_log)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title("Taskspace tracking performance : {} m/s (simulation)".format(testing_speed))
    ax.plot(traj_info[:,0,0], traj_info[:,0,1], traj_info[:,0,2], label='ROM', c='g')
    ax.plot(traj_info[:,1,0], traj_info[:,1,1], traj_info[:,1,2], c='g')
    ax.plot(traj_info[:,2,0], traj_info[:,2,1], traj_info[:,2,2], c='g')
    ax.plot(actual_state_info[:,0,0], actual_state_info[:,0,1], actual_state_info[:,0,2], label='robot', c='r')
    ax.plot(actual_state_info[:,1,0], actual_state_info[:,1,1], actual_state_info[:,1,2], c='r')
    ax.plot(actual_state_info[:,2,0], actual_state_info[:,2,1], actual_state_info[:,2,2], c='r')

    set_axes_equal(ax)

    plt.legend()
    plt.tight_layout()
    plt.savefig("./plots/taskspace{}.png".format(testing_speed))

    traj_info = traj_info.reshape(-1, 9)
    actual_state_info = actual_state_info.reshape(-1, 9)
    time_log = time_log.reshape(-1, 1)
    l_footstep = np.array(l_footstep)
    r_footstep = np.array(r_footstep)

    x_error = np.linalg.norm(traj_info[:,0] - actual_state_info[:,0])
    y_error = np.linalg.norm(traj_info[:,1] - actual_state_info[:,1])
    z_error = np.linalg.norm(traj_info[:,2] - actual_state_info[:,2])

    # print(traj_info.shape)
    # print(actual_state_info.shape)
    # print(time_log.shape)

    # return matrix of logged data
    return np.array([x_error, y_error, z_error]), np.array([l_footstep, r_footstep])

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="../../trained_models/ppo/Cassie-v0/IK_traj-aslip_aslip_joint_2048_12288_seed-10/", help="path to folder containing policy and run details")
parser.add_argument("--traj_len", default=100, type=str)  # timesteps once at speed to collect data
parser.add_argument("--ramp_up", default=100, type=str)  # timesteps for coming up to speed, before data collection starts
parser.add_argument("--debug", default=False, action='store_true')
parser.add_argument("--viz", default=False, action='store_true')
# parser.add_argument("--eval", default=True, action="store_false", help="Whether to call policy.eval() or not")

args = parser.parse_args()

run_args = pickle.load(open(args.path + "experiment.pkl", "rb"))

print(args.path)

policy = torch.load(args.path + "actor.pt")

# if args.eval:
#     policy.eval()  # NOTE: for some reason the saved nodelta_neutral_stateest_symmetry policy needs this but it breaks all new policies...
# policy.eval()

data = []
footdata = []

speeds = [i/10 for i in range(21)] # 0.0 to 2.0 m/s
# speeds = [i/10 for i in range(3, 4)] # 0.0 to 2.0 m/s
for speed in speeds:
    taskspace_data, footplacement_data = eval_policy(policy, args, run_args, speed)
    data.append(taskspace_data)
    footdata.append(footplacement_data)

data = np.array(data)
footdata = np.array(footdata)
print(data.shape)
print(footdata.shape)
# Center of mass position tracking error
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
colors = ["tab:blue", "tab:red", "tab:green"]
for i in range(data.shape[0]):
    x_error = data[i, 0]
    y_error = data[i, 1]
    z_error = data[i, 2]
    if i == 0:
        ax.bar(i, z_error, label='z', bottom=x_error+y_error, color=colors[0])
        ax.bar(i, y_error, label='y', bottom=x_error, color=colors[1])
        ax.bar(i, x_error, label='x', color=colors[2])
    else:
        ax.bar(i, z_error, bottom=x_error+y_error, color=colors[0])
        ax.bar(i, y_error, bottom=x_error, color=colors[1])
        ax.bar(i, x_error, color=colors[2])
ax.set_title('Average COM Tracking Error')
ax.set_ylabel('Avg. Error (cm)')
ax.set_xticks(np.arange(len(speeds)))
ax.set_xticklabels([str(speed) for speed in speeds])
plt.legend()
plt.savefig("./plots/compos_err.png")

# # Foot Placement tracking error
# fig2 = plt.figure(figsize=(10,10))
# ax2 = fig2.add_subplot(111)
# colors = ["tab:blue", "tab:red"]
# for i in range(footdata.shape[0]):
#     x_error = np.mean(footdata[i, 0])
#     y_error = np.mean(footdata[i, 1])
#     # only label once
#     if i == 0:
#         ax2.bar(i, y_error, label='y', bottom=x_error, color=colors[0])
#         ax2.bar(i, x_error, label='x', color=colors[1])
#     else:
#         ax2.bar(i, y_error, bottom=x_error, color=colors[0])
#         ax2.bar(i, x_error, color=colors[1])
# ax2.set_title('Average Foot Placement Error')
# ax2.set_ylabel('Avg. Error (cm)')
# ax2.set_xticks(np.arange(len(speeds)))
# ax2.set_xticklabels([str(speed) for speed in speeds])
# plt.legend()
# plt.savefig("./plots/footpos_err.png")


# plt.show()
