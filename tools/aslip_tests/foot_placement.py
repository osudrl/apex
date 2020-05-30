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

def eval_policy(policy, args, run_args, testing_speed):

    max_traj_len = args.traj_len
    visualize = not args.no_viz
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

    while footstep_count-1 < 15:
    
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

        # print("{} : {} : {}".format(env.phase, env.l_foot_pos[2], env.r_foot_pos[2]))
        # allow some ramp up time
        while timesteps > env.phaselen * 2:

            # fill r_last_ideal_pos and l_last_ideal_pos with values before collecting data
                        
            # check if we are in left swing phase
            # left foot swing, right foot stance
            if env.phase == 7:
                left_swing = True
                right_swing = False
                r_ideal_land_pos = env.r_foot_pos[:2] + get_ref_aslip_global_state_no_drift_correct(env, phase=7)[2][:2]
                r_foot_poses_ideal.append(r_ideal_land_pos)
                # print("left-swing : right should land at {}".format(r_ideal_land_pos))
            # left foot stance, right foot swing
            elif env.phase == 23:
                left_swing = False
                right_swing = True
                l_ideal_land_pos = env.l_foot_pos[:2] + get_ref_aslip_global_state_no_drift_correct(env, phase=23)[0][:2]
                l_foot_poses_ideal.append(l_ideal_land_pos)
                # print("right-swing : left should land at {}".format(l_ideal_land_pos))
                footstep_count += 1
            else:
                break

            # if left foot is down (right swinging) calculate error between it and ideal position, calculate next ideal position.
            # if right foot is down (left swinging) "---------------------------------------------------------------------------"
            if footstep_count >= 1:

                if l_ideal_land_pos is not None and left_swing == False:
                    l_actual_land_pos = env.l_foot_pos[:2]
                    l_foot_poses_actual.append(l_ideal_land_pos)
                    footplace_err.append(np.linalg.norm(l_ideal_land_pos - l_actual_land_pos))
                    # print("left landed at {}".format(l_actual_land_pos))
                elif r_ideal_land_pos is not None and right_swing == False:
                    r_actual_land_pos = env.r_foot_pos[:2]
                    r_foot_poses_actual.append(r_ideal_land_pos)
                    footplace_err.append(np.linalg.norm(r_ideal_land_pos - r_actual_land_pos))
                    # print("right landed at {}".format(r_actual_land_pos))
            
            break

        # if a[1][2] == 0.0:
        #     l_footstep.append(np.linalg.norm(a[1] - d[1]))
        # elif a[2][2] == 0.0:
        #     r_footstep.append(np.linalg.norm(a[2] - d[2]))

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

    print("Eval reward: ", eval_reward)

    l_foot_poses_ideal = np.array(l_foot_poses_ideal)
    r_foot_poses_ideal = np.array(r_foot_poses_ideal)
    l_foot_poses_actual = np.array(l_foot_poses_actual)
    r_foot_poses_actual = np.array(r_foot_poses_actual)
    footplace_err = np.array(footplace_err)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    ax.scatter(l_foot_poses_ideal[:, 0], l_foot_poses_ideal[:, 1], c='tab:green')
    ax.scatter(r_foot_poses_ideal[:, 0], r_foot_poses_ideal[:, 1], c='tab:green')
    ax.scatter(l_foot_poses_actual[:, 0], l_foot_poses_actual[:, 1], c='tab:red')
    ax.scatter(r_foot_poses_actual[:, 0], r_foot_poses_actual[:, 1], c='tab:red')
    ax.axis('equal')
    plt.savefig('./plots/footplace{}.png'.format(testing_speed))
    # plt.show()

    return footplace_err


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="../../trained_models/ppo/Cassie-v0/IK_traj-aslip_aslip_old_2048_12288_seed-10/", help="path to folder containing policy and run details")
parser.add_argument("--traj_len", default=400, type=str)
parser.add_argument("--debug", default=False, action='store_true')
parser.add_argument("--no_viz", default=False, action='store_true')
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
    data.append(eval_policy(policy, args, run_args, speed))
data = np.array(data)

# Foot Placement tracking error
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
colors = ["tab:blue", "tab:red"]
for i in range(data.shape[0]):
    error = np.mean(data[i])
    ax.bar(i, error, color="tab:blue")
ax.set_title('Average Foot Placement Error')
ax.set_ylabel('Avg. Error (cm)')
ax.set_xticks(np.arange(len(speeds)))
ax.set_xticklabels([str(speed) for speed in speeds])
plt.savefig("./plots/footpos_err.png")
