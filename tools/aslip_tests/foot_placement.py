"""
Measures the taskspace tracking error for aslip policies individually for each speed the policy accepts
"""

import os, sys, argparse
sys.path.append("../..") 

from cassie.trajectory import get_ref_aslip_global_state_no_drift_correct
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

def eval_policy(policy, args, run_args, testing_speed, num_steps=10):

    max_traj_len = args.traj_len
    visualize = not args.no_vis
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
    left_swing, right_swing, l_ideal_land_pos, r_ideal_land_pos = False, False, None, None
    l_foot_poses_actual = []
    l_foot_poses_ideal = []
    r_foot_poses_actual = []
    r_foot_poses_ideal = []
    footplace_err = []
    footstep_count = 0

    env.update_speed(testing_speed)
    print(env.speed)

    ## Get the fixed delta for each footstep position change
    right_delta = get_ref_aslip_global_state_no_drift_correct(env, phase=7)

    while footstep_count-1 < num_steps:
    
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

        # get the delta for right and left by looking at phases where both feet are in stance
        right_to_left = get_ref_aslip_global_state_no_drift_correct(env, phase=16)[2][:2] - get_ref_aslip_global_state_no_drift_correct(env, phase=16)[0][:2]
        left_to_right = get_ref_aslip_global_state_no_drift_correct(env, phase=29)[0][:2] - get_ref_aslip_global_state_no_drift_correct(env, phase=29)[2][:2]
        delta = right_to_left + left_to_right
        # print(right_to_left)
        # print(left_to_right)
        # print(delta)
        # print(get_ref_aslip_global_state_no_drift_correct(env, phase=16)[2][:2])
        # print(get_ref_aslip_global_state_no_drift_correct(env, phase=16)[2][:2] + delta)
        # exit()

        # print("{} : {} : {}".format(env.phase, env.l_foot_pos[2], env.r_foot_pos[2]))
        # allow some ramp up time
        if timesteps > env.phaselen * 2:
            # fill r_last_ideal_pos and l_last_ideal_pos with values before collecting data

            # check if we are in left swing phase
            # left foot swing, right foot stance
            if env.phase == 7:
                left_swing = True
                right_swing = False
                r_actual_land_pos = env.r_foot_pos[:2]
                # if we have data from last step, we can calculate error from where right foot landed to where it should have landed, BEFORE updating r_ideal_land_pos
                if footstep_count >= 1:
                    r_foot_poses_actual.append(r_actual_land_pos)
                    footplace_err.append(np.linalg.norm(r_ideal_land_pos - r_actual_land_pos))
                    # print("right landed at\t\t\t{}".format(r_actual_land_pos))
                l_ideal_land_pos = r_actual_land_pos + right_to_left
                l_foot_poses_ideal.append(l_ideal_land_pos)
                # print("next, left should land at\t{}\t{}\n".format(l_ideal_land_pos, r_actual_land_pos))
                # print("left-swing : right should land at {}".format(r_ideal_land_pos))
            # left foot stance, right foot swing
            elif env.phase == 23:
                left_swing = False
                right_swing = True
                l_actual_land_pos = env.l_foot_pos[:2]
                # if we have data from last step, we can calculate error from where right foot landed to where it should have landed, BEFORE updating r_ideal_land_pos
                if footstep_count >= 1:
                    l_foot_poses_actual.append(l_actual_land_pos)
                    footplace_err.append(np.linalg.norm(l_ideal_land_pos - l_actual_land_pos))
                    # print("left landed at\t\t\t{}".format(l_actual_land_pos))
                r_ideal_land_pos = l_actual_land_pos + left_to_right
                r_foot_poses_ideal.append(r_ideal_land_pos)
                # print("next, right should land at\t{}\t{}\n".format(r_ideal_land_pos, l_actual_land_pos))
                footstep_count += 1
                # print("right-swing : left should land at {}".format(l_ideal_land_pos))

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

    print("Avg Foot Placement Error: ", np.mean(footplace_err))
    print("Num pairs: ", len(footplace_err))

    # trim the ideal footplace poses
    l_foot_poses_ideal = l_foot_poses_ideal[1:]
    r_foot_poses_ideal = r_foot_poses_ideal[:-1]

    l_foot_poses_ideal = np.array(l_foot_poses_ideal)
    r_foot_poses_ideal = np.array(r_foot_poses_ideal)
    l_foot_poses_actual = np.array(l_foot_poses_actual)
    r_foot_poses_actual = np.array(r_foot_poses_actual)
    footplace_err = np.array(footplace_err)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    ax.scatter(l_foot_poses_ideal[:, 0], l_foot_poses_ideal[:, 1], c='tab:green', label='desired', alpha=0.5)
    ax.scatter(r_foot_poses_ideal[:, 0], r_foot_poses_ideal[:, 1], c='tab:green', alpha=0.5)
    ax.scatter(l_foot_poses_actual[:, 0], l_foot_poses_actual[:, 1], c='tab:red', label='actual', alpha=0.5)
    ax.scatter(r_foot_poses_actual[:, 0], r_foot_poses_actual[:, 1], c='tab:red', alpha=0.5)
    ax.axis('equal')
    ax.set_ylabel('y (m)')
    ax.set_xlabel('x (m)')
    ax.set_title('Desired vs Actual Foot Placements ({} m/s)'.format(testing_speed))
    plt.savefig('./plots/footplace{}.png'.format(testing_speed))
    # plt.show()

    return footplace_err


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="../../trained_models/ppo/Cassie-v0/IK_traj-aslip_aslip_old_2048_12288_seed-10/", help="path to folder containing policy and run details")
parser.add_argument("--traj_len", default=400, type=str)
parser.add_argument("--num_steps", default=10, type=str)
parser.add_argument("--debug", default=False, action='store_true')
parser.add_argument("--no_vis", default=False, action='store_true')
parser.add_argument("--eval", default=True, action="store_false", help="Whether to call policy.eval() or not")

args = parser.parse_args()

run_args = pickle.load(open(args.path + "experiment.pkl", "rb"))

policy = torch.load(args.path + "actor.pt")

if args.eval:
    policy.eval()  # NOTE: for some reason the saved nodelta_neutral_stateest_symmetry policy needs this but it breaks all new policies...

data = []
speeds = [i/10 for i in range(0,21)] # 0.0 to 2.0 m/s
# speeds = [i/10 for i in range(3, 4)] # 0.0 to 2.0 m/s
for speed in speeds:
    data.append(eval_policy(policy, args, run_args, speed, num_steps=args.num_steps))
data = np.array(data)

# Foot Placement tracking error
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
colors = ["tab:blue", "tab:red"]
for i in range(data.shape[0]):
    error = np.mean(data[i])
    ax.bar(i, error, color="tab:blue")
ax.set_title('Average Foot Placement Error')
ax.set_ylabel('Avg. Error (m)')
ax.set_xticks(np.arange(len(speeds)))
ax.set_xticklabels([str(speed) for speed in speeds])
plt.savefig("./plots/footpos_err.png")
