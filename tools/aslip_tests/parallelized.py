"""
Measures the taskspace tracking error for aslip policies individually for each speed the policy accepts
"""

import os, sys, argparse

from functools import partial
from cassie.trajectory import get_ref_aslip_global_state_no_drift_correct
# from cassie import CassieEnv, CassiePlayground
from rl.policies.actor import GaussianMLP_Actor

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pickle
import numpy as np
import torch
import time

import ray

from util import env_factory

@ray.remote
def footstep_test(policy, env_fn, testing_speed, num_trials, num_steps, vis=False):
    
    if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()

    all_l_foot_poses_actual = []
    all_l_foot_poses_ideal = []
    all_r_foot_poses_actual = []
    all_r_foot_poses_ideal = []
    all_footplace_err = []

    for trial_num in range(num_trials):

        env = env_fn()

        if vis:
            env.render()

        l_foot_poses_actual = []
        l_foot_poses_ideal = []
        r_foot_poses_actual = []
        r_foot_poses_ideal = []
        footplace_err = []

        state = env.reset_for_test()
        done = False
        timesteps = 0
        eval_reward = 0

        # Data to track
        left_swing, right_swing, l_ideal_land_pos, r_ideal_land_pos = False, False, None, None
        footstep_count = 0

        env.update_speed(testing_speed)
        # print(env.speed)

        ## Get the fixed delta for each footstep position change
        right_delta = get_ref_aslip_global_state_no_drift_correct(env, phase=7)

        while footstep_count-1 < num_steps:
        
            if hasattr(env, 'simrate'):
                start = time.time()
                
            action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()

            state, reward, done, _ = env.step(action)

            # get the delta for right and left by looking at phases where both feet are in stance
            right_to_left = get_ref_aslip_global_state_no_drift_correct(env, phase=16)[2][:2] - get_ref_aslip_global_state_no_drift_correct(env, phase=16)[0][:2]
            left_to_right = get_ref_aslip_global_state_no_drift_correct(env, phase=29)[0][:2] - get_ref_aslip_global_state_no_drift_correct(env, phase=29)[2][:2]
            delta = right_to_left + left_to_right

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

            if vis:
                render_state = env.render()
            # if hasattr(env, 'simrate'):
            #     # assume 40hz
            #     end = time.time()
            #     delaytime = max(0, 1000 / 40000 - (end-start))
            #     time.sleep(delaytime)

        # trim the ideal footplace poses
        l_foot_poses_ideal = l_foot_poses_ideal[1:]
        r_foot_poses_ideal = r_foot_poses_ideal[:-1]

        all_l_foot_poses_actual.extend(l_foot_poses_actual)
        all_l_foot_poses_ideal.extend(l_foot_poses_ideal)
        all_r_foot_poses_actual.extend(r_foot_poses_actual)
        all_r_foot_poses_ideal.extend(r_foot_poses_ideal)
        all_footplace_err.append(footplace_err)

    all_l_foot_poses_ideal = np.array(all_l_foot_poses_ideal)
    all_r_foot_poses_ideal = np.array(all_r_foot_poses_ideal)
    all_l_foot_poses_actual = np.array(all_l_foot_poses_actual)
    all_r_foot_poses_actual = np.array(all_r_foot_poses_actual)
    all_footplace_err = np.array(all_footplace_err)

    print(all_footplace_err.shape)

    return all_l_foot_poses_ideal, all_r_foot_poses_ideal, all_l_foot_poses_actual, all_r_foot_poses_actual, all_footplace_err

speeds = [i/10 for i in range(0,21)]

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_procs", default=20, type=int)
    parser.add_argument("--traj_len", default=400, type=int)
    parser.add_argument("--num_steps", default=10, type=int)
    parser.add_argument("--num_trials", default=5, type=int)
    parser.add_argument("--debug", default=False, action='store_true')
    parser.add_argument("--no_vis", default=False, action='store_true')
    parser.add_argument("--eval", default=True, action="store_false", help="Whether to call policy.eval() or not")
    args = parser.parse_args()

    ray.init(num_cpus=args.num_procs)

    paths = ["./trained_models/ppo/Cassie-v0/traj-aslip_aslip_old_2048_12288_seed-0/",
            "./trained_models/ppo/Cassie-v0/traj-aslip_aslip_old_2048_12288_seed-10/",
            "./trained_models/ppo/Cassie-v0/traj-aslip_aslip_old_2048_12288_seed-20/",
            "./trained_models/ppo/Cassie-v0/traj-aslip_aslip_old_2048_12288_seed-30/"]

    data = []

    ideal_foot_poses = []
    actual_foot_poses = []
    placement_errors = []

    for path in paths:

        # Get policy, create env constructor
        run_args = pickle.load(open(path + "experiment.pkl", "rb"))
        policy = torch.load(path + "actor.pt")

        env_fn = env_factory("Cassie-v0", traj="aslip", state_est=run_args.state_est, no_delta=run_args.no_delta, learn_gains=run_args.learn_gains, ik_baseline=run_args.ik_baseline, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward="aslip_old", history=run_args.history)

        # parallelized loop for speed
        data_id = [footstep_test.remote(policy, env_fn, speed, args.num_trials, args.num_steps) for speed in speeds]

        foo = ray.get(data_id)

        data.append(foo)

    data = np.array(data)
    print(data.shape)

    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)



## Just vis
def vis():
    import matplotlib

    font = {'family' : 'serif',
            'size'   : 12}

    matplotlib.rc('font', **font)

    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)

    for speed_idx in range(data.shape[1]):

        # Foot Placement Scatterplot (combined)
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        for pol_idx in range(data.shape[0]):
            ax.scatter(data[pol_idx, speed_idx, 0][:, 0], data[pol_idx, speed_idx, 0][:, 1], c="tab:green", label="desired", alpha=0.5)
            ax.scatter(data[pol_idx, speed_idx, 1][:, 0], data[pol_idx, speed_idx, 1][:, 1], c="tab:green", alpha=0.5)
            ax.scatter(data[pol_idx, speed_idx, 2][:, 0], data[pol_idx, speed_idx, 2][:, 1], c="tab:red", label="actual", alpha=0.5)
            ax.scatter(data[pol_idx, speed_idx, 3][:, 0], data[pol_idx, speed_idx, 3][:, 1], c="tab:red", alpha=0.5)
        ax.axis('equal')
        ax.set_ylabel('y (m)')
        ax.set_xlabel('x (m)')
        ax.set_title('Desired vs Actual Foot Placements ({} m/s)'.format(speeds[speed_idx]))
        plt.savefig('./plots/footplace{}.png'.format(speeds[speed_idx]))

    # Error Bar Chart ( Combined )
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111)
    for speed_idx in range(data.shape[1]):
        err_data = np.array([data[i,speed_idx, 4] for i in range(data.shape[0])])
        err_data = err_data.reshape(-1, err_data.shape[-1])
        # print(err_data.shape)
        error, err_range = np.mean(err_data), np.std(err_data)
        ax.bar(speed_idx, error, yerr=err_range, capsize=3, color='dodgerblue')
    plt.xticks(np.arange(len(speeds)), [str(speed) for speed in speeds], rotation=45)
    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % 2 != 0:
            label.set_visible(False)
    ax.set_ylabel('Avg. Error (cm)')
    ax.set_xlabel('Commanded Speed (m/s)')
    plt.savefig("./plots/footpos_err.svg")
    plt.savefig("./plots/footpos_err.pdf")
    plt.savefig("./plots/footpos_err.png")

    # Error Bar Chart ( For each policy )
    fig, axs = plt.subplots(data.shape[0], 1, sharex=False, figsize=(10,10))
    for i, ax in enumerate(axs):
        for speed_idx in range(data.shape[1]):
            err_data = np.array(data[i,speed_idx, 4])
            err_data = err_data.reshape(-1, err_data.shape[-1])
            # print(err_data.shape)
            error, err_range = np.mean(err_data), np.std(err_data)
            ax.bar(speed_idx, error, yerr=err_range, capsize=3, color='dodgerblue')
        plt.xticks(np.arange(len(speeds)), [str(speed/10) for speed in speeds], rotation=45)
        for index, label in enumerate(ax.xaxis.get_ticklabels()):
            if index % 2 != 0:
                label.set_visible(False)
        ax.set_ylabel('Avg. Error (m)')
        ax.set_xlabel('Commanded Speed (m/s)')
    plt.tight_layout()
    plt.savefig("./plots/footpos_err_separate.png")
    


#     print(len(foo))

#     print(all_l_foot_poses_ideal.shape)
#     print(all_r_foot_poses_ideal.shape)
#     print(all_l_foot_poses_actual.shape)
#     print(all_r_foot_poses_actual.shape)
#     print(all_footplace_err.shape)

#     ideal_foot_poses.append([all_l_foot_poses_ideal, all_r_foot_poses_ideal])
#     actual_foot_poses.append([all_l_foot_poses_actual, all_r_foot_poses_actual])
#     placement_errors.append(all_footplace_err)

# print(ideal_foot_poses.shape)
# print(actual_foot_poses.shape)
# print(placement_errors.shape)

# main()
vis()