import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from torch.autograd import Variable
import time, os, sys
import math

from util import env_factory
from rl.policies.actor import GaussianMLP_Actor

import argparse
import pickle

def collect_data():
    seeds = [0, 10, 20]
    wait_cycles = 3
    num_cycles = 10
    speed = 1.0

    # Make envs
    aslip_path = "./trained_models/comparison/aslip_delta_policies/traj-aslip_aslip_old_2048_12288_seed-{}".format(seeds[0])
    iros_path = "./trained_models/comparison/iros_retrain_policies/clock_traj-walking_iros_paper_2048_12288_seed-{}".format(seeds[0])
    aslip_args = pickle.load(open(os.path.join(aslip_path, "experiment.pkl"), "rb"))
    aslip_env_fn = env_factory(aslip_args.env_name, traj=aslip_args.traj, state_est=aslip_args.state_est, no_delta=aslip_args.no_delta, dynamics_randomization=aslip_args.dyn_random, 
                        mirror=False, clock_based=aslip_args.clock_based, reward="iros_paper", history=aslip_args.history)
    aslip_env = aslip_env_fn()
    iros_args = pickle.load(open(os.path.join(iros_path, "experiment.pkl"), "rb"))
    iros_env_fn = env_factory(iros_args.env_name, traj=iros_args.traj, state_est=iros_args.state_est, no_delta=iros_args.no_delta, dynamics_randomization=iros_args.dyn_random, 
                        mirror=False, clock_based=iros_args.clock_based, reward="iros_paper", history=iros_args.history)
    iros_env = iros_env_fn()

    aslip_state = torch.Tensor(aslip_env.reset_for_test())
    iros_state = torch.Tensor(iros_env.reset_for_test())
    aslip_env.update_speed(speed)
    iros_env.speed = speed
    aslip_phaselen = aslip_env.phaselen + 1
    iros_phaselen = iros_env.phaselen + 1

    aslip_data = np.zeros((len(seeds), num_cycles, aslip_env.simrate*(aslip_phaselen), 2))
    iros_data = np.zeros((len(seeds), num_cycles, iros_env.simrate*(iros_phaselen), 2))
    for s in range(len(seeds)):
        print("running seed {}".format(seeds[s]))
        aslip_path = "./trained_models/comparison/aslip_delta_policies/traj-aslip_aslip_old_2048_12288_seed-{}".format(seeds[s])
        iros_path = "./trained_models/comparison/iros_retrain_policies/clock_traj-walking_iros_paper_2048_12288_seed-{}".format(seeds[s])

        # Load policies
        aslip_policy = torch.load(os.path.join(aslip_path, "actor.pt"))
        iros_policy = torch.load(os.path.join(iros_path, "actor.pt"))

        # print("iros: ", iros_env.simrate, iros_env.phaselen)
        # print("aslip: ", aslip_env.simrate, aslip_env.phaselen)

        with torch.no_grad():
            # Run few cycles to stabilize (do separate incase two envs have diff phaselens)
            for i in range(wait_cycles*(aslip_phaselen)):
                action = aslip_policy.forward(torch.Tensor(aslip_state), deterministic=True).detach().numpy()
                aslip_state, reward, done, _ = aslip_env.step(action)
                aslip_state = torch.Tensor(aslip_state)
                # curr_qpos = aslip_env.sim.qpos()
                # print("curr height: ", curr_qpos[2])
            for i in range(wait_cycles*(iros_phaselen)):
                action = iros_policy.forward(torch.Tensor(iros_state), deterministic=True).detach().numpy()
                iros_state, reward, done, _ = iros_env.step(action)
                iros_state = torch.Tensor(iros_state)

            # Collect actual data
            print("Start actual data")
            for i in range(num_cycles):
                for j in range(aslip_phaselen):
                    action = aslip_policy.forward(torch.Tensor(aslip_state), deterministic=True).detach().numpy()
                    for k in range(aslip_env.simrate):
                        aslip_env.step_simulation(action)
                        aslip_data[s, i, j*aslip_env.simrate + k, :] = aslip_env.sim.get_foot_forces()
                    
                    aslip_env.time  += 1
                    aslip_env.phase += aslip_env.phase_add
                    if aslip_env.phase > aslip_env.phaselen:
                        aslip_env.phase = 0
                        aslip_env.counter += 1
                    aslip_state = aslip_env.get_full_state()

            for i in range(num_cycles):
                for j in range(iros_phaselen):
                    action = iros_policy.forward(torch.Tensor(iros_state), deterministic=True).detach().numpy()
                    for k in range(iros_env.simrate):
                        iros_env.step_simulation(action)
                        iros_data[s, i, j*iros_env.simrate + k, :] = iros_env.sim.get_foot_forces()
                    
                    iros_env.time  += 1
                    iros_env.phase += iros_env.phase_add
                    if iros_env.phase > iros_env.phaselen:
                        iros_env.phase = 0
                        iros_env.counter += 1
                    iros_state = iros_env.get_full_state()
    
    np.save("./trained_models/comparison/aslip_delta_policies/avg_GRFs_speed{}".format(speed), aslip_data)
    np.save("./trained_models/comparison/iros_retrain_policies/avg_GRFs_speed{}".format(speed), iros_data)
        
def plot_data():
    speeds = [0.0, 1.0, 2.0]
    text_size = 20
    title_size = 24
    mpl.rcParams['font.family'] = "Serif"
    fig, ax = plt.subplots(len(speeds), 2, figsize=(10, 12), constrained_layout=True)
    fig.suptitle("Ground Reaction Force Comparison for Different Speeds", fontsize=title_size)

    # with open("./GRF_2KHz.pkl", "rb") as f:
        # total_model_data = pickle.load(f)
    total_model_GRFs = pickle.load(open("./GRF_2KHz.pkl", "rb"))
    speed_inds = [i/10 for i in range(0, 21)]
    

    ymin = 0
    ymax = 0
    for i in range(len(speeds)):
        aslip_data = np.load("./trained_models/comparison/aslip_delta_policies/avg_GRFs_speed{}.npy".format(speeds[i]))
        iros_data = np.load("./trained_models/comparison/iros_retrain_policies/avg_GRFs_speed{}.npy".format(speeds[i]))
        model_GRFs = total_model_GRFs[speed_inds.index(speeds[i])][:]
        # print(np.max(model_GRFs[0]))
        # exit()

        # Average data and get std dev
        # mean_aslip = np.mean(aslip_data, axis=(0, 1))
        mean_aslip = np.mean(aslip_data, axis=(1))
        mean_aslip = mean_aslip[0, :, :]
        # print(mean_aslip.shape)
        # exit()
        # stddev_aslip = np.std(aslip_data, axis=(0, 1))
        stddev_aslip = np.std(aslip_data, axis=(1))
        stddev_aslip = stddev_aslip[0, :, :]
        mean_iros = np.mean(iros_data, axis=(0, 1))
        stddev_iros = np.std(iros_data, axis=(0, 1))
        print(mean_aslip.shape)
        print(mean_iros.shape)

        total_data = [mean_aslip, mean_iros]
        total_stddev = [stddev_aslip, stddev_iros]

        max_time = max(mean_aslip.shape[0], mean_iros.shape[0], model_GRFs[0][-1])
        xticks = np.arange(0, max_time*0.0005, .4)
        
        ax[i][0].plot(model_GRFs[0], model_GRFs[2], '--', label="Reference Left", color="tab:cyan", )
        ax[i][0].plot(model_GRFs[0], model_GRFs[1], '--', label="Reference Right", color="tab:red", )
        for j in range(2):
            curr_data = total_data[j]
            curr_stddev = total_stddev[j]
            fill_minus = curr_data - curr_stddev
            fill_plus = curr_data + curr_stddev
            if ymin > np.min(fill_minus):
                ymin = np.min(fill_minus)
            if ymax < np.max(fill_plus):
                ymax = np.max(fill_plus)
            time = np.linspace(0, curr_data.shape[0]*0.0005, curr_data.shape[0])
            ax[i][j].plot(time, curr_data[:, 0], label="Sim Left Foot", color="tab:cyan")
            ax[i][j].plot(time, curr_data[:, 1], label="Sim Right Foot", color="tab:red")
            ax[i][j].fill_between(time, fill_minus[:, 0], fill_plus[:, 0], color="tab:cyan", alpha=.3, edgecolor=None)
            ax[i][j].fill_between(time, fill_minus[:, 1], fill_plus[:, 1], color="tab:red", alpha=.3, edgecolor=None)
            ax[i][j].legend(loc="upper right")
            # ax[i][j].set_title("{}, Speed = {}".format(titles[j], speeds[i]), fontsize=text_size)
            ax[i][j].set_xlim(0, xticks[-1] + 0.05)
            ax[i][j].set_xticks(xticks)
        
        ax[i][0].set_ylabel("GRFs (N) (Speed={})".format(speeds[i]), fontsize=text_size)
    # Column only labels
    titles = ["ASLIP Policy", "Xie et. al. Method"]
    for i in range(2):
        ax[2][i].set_xlabel("Time (s)", fontsize=text_size)
        ax[0][i].set_title(titles[i], fontsize=text_size)
    # Reset y limits and ticks for all plots
    print("ymin: {}\t ymax: {}".format(ymin, ymax))
    ymin = int(math.floor(ymin / 50))*50
    ymax = int(math.ceil(ymax / 50))*50
    print("ymin: {}\t ymax: {}".format(ymin, ymax))
    yticks = np.arange(ymin, ymax, 200)
    for i in range(len(speeds)):

        for j in range(2):
            if (i != 2):
                # pass
                # ax[i][j].spines['bottom'].set_visible(False)
                ax[i][j].set_xticks([])
            ax[i][j].set_ylim(ymin, ymax)
            if (j == 1):
                ax[i][1].spines['left'].set_visible(False)
                ax[i][1].set_yticks([])
            else:
                ax[i][j].set_yticks(yticks)
            ax[i][j].spines['right'].set_visible(False)
            ax[i][j].spines['top'].set_visible(False)
            for tick in ax[i][j].xaxis.get_major_ticks():
                tick.label.set_fontsize(text_size)
            for tick in ax[i][j].yaxis.get_major_ticks():
                tick.label.set_fontsize(text_size)
    

    # plt.tight_layout()
    # plt.show()
    plt.savefig("./GRF_compare_single_aslip.png")

# collect_data()
plot_data()

    