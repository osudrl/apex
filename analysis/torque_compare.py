import numpy as np
import matplotlib.pyplot as plt
import time, os, sys

import argparse
import pickle

sim_data = np.load("./sim_torques_fast_375.npz")
hardware_data = np.load("./hardware_torques_fast_375.npz")
sim_torques = sim_data["torques"]
sim_time = sim_data["time"]
sim_motors = sim_data["motors"]
sim_motor_pos = sim_data["motors"]
sim_nn = sim_data["actions"]
sim_targets = sim_data["target"]
sim_states = sim_data["states"]
sim_time_lf = sim_data["time_lf"]
sim_motor_vel = sim_data["motor_vel"]
hardware_torques = hardware_data["torques"]
hardware_time = hardware_data["time"]
hardware_time -= hardware_time[0]
hardware_time_lf = hardware_data["time_lf"]
hardware_time_lf -= hardware_time_lf[0]
hardware_motors = hardware_data["motors"]
hardware_motor_pos = hardware_data["motors"]
hardware_nn = hardware_data["actions"]
hardware_targets = hardware_data["targets"]
hardware_states = hardware_data["states"]
hardware_motor_vel = hardware_data["motor_vel"]
print(hardware_time_lf.shape)

sim_motors_diff = sim_motors[1:] - sim_motors[0:-1]
hardware_motors_diff = hardware_motors[1:] - hardware_motors[0:-1]
sim_len = sim_torques.shape[0]
hard_len = hardware_torques.shape[0]
orig_plot_len = min(sim_len, hard_len)
plot_len = orig_plot_len
plot_len_lf = min(sim_time_lf.shape[0], hardware_time_lf.shape[0])

do_save = False
save_dir = "./torque_compare_plots/"

vis = { "mpos_input": False,
        "mvel_input": False,
        "torques": True,
        "mpos_diff": False,
        "torque_mpos_diff": False,
        "torque_pos_error": False,
        "torque_pos_error_knee": False,
        "footpos": False,
        "pel_vel": False,
        "targets": True,
        "pos_error": False,
        "footpos": False,
        "mvel": False,
        "mpos": False, 
        "input": False}

# Graph input data
if vis["input"]:
    fig, ax = plt.subplots(2, 3, figsize=(15, 5))
    titles = ["X", "Y", "Z"]
    ax[0][0].set_ylabel("Position")
    ax[1][0].set_ylabel("Position")
    for i in range(3):
       
        ax[0][i].plot(sim_time_lf[:plot_len_lf], sim_states[:plot_len_lf, i], label="sim")
        ax[0][i].plot(hardware_time_lf[:plot_len_lf], hardware_states[:plot_len_lf, i], label="hardware")
        ax[0][i].set_title("Left " + titles[i])
        ax[0][i].legend()

        ax[1][i].plot(sim_time_lf[:plot_len_lf], sim_states[:plot_len_lf, i+3], label="sim")
        ax[1][i].plot(hardware_time_lf[:plot_len_lf], hardware_states[:plot_len_lf, i+3], label="hardware")
        ax[1][i].set_title("Right " + titles[i])
        ax[1][i].set_xlabel("Time")
        ax[1][i].legend()

    fig.suptitle("Input Foot Pos")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if do_save:
        plt.savefig(os.path.join(save_dir, "foot_pos_in.png"))
    else:
        plt.show()

    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    titles = ["W", "X", "Y", "Z"]
    ax[0].set_ylabel("Quat")
    for i in range(4):
       
        ax[i].plot(sim_time_lf[:plot_len_lf], sim_states[:plot_len_lf, 6+i], label="sim")
        ax[i].plot(hardware_time_lf[:plot_len_lf], hardware_states[:plot_len_lf, 6+i], label="hardware")
        ax[i].set_title("Left " + titles[i])
        ax[i].legend()

    fig.suptitle("Input Pel Quat")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if do_save:
        plt.savefig(os.path.join(save_dir, "foot_pos_in.png"))
    else:
        plt.show()
        
    fig, ax = plt.subplots(2, 5, figsize=(15, 5))
    # t = np.linspace(0, num_steps-1, num_steps*simrate)
    titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
    ax[0][0].set_ylabel("Position")
    ax[1][0].set_ylabel("Position")
    for i in range(5):
        
        ax[0][i].plot(sim_time_lf[:plot_len_lf], sim_states[:plot_len_lf, 10+i], label="sim")
        ax[0][i].plot(hardware_time_lf[:plot_len_lf], hardware_states[:plot_len_lf, 10+i], label="hardware")
        ax[0][i].set_title("Left " + titles[i])
        ax[0][i].legend()

        ax[1][i].plot(sim_time_lf[:plot_len_lf], sim_states[:plot_len_lf, 10+i+5], label="sim")
        ax[1][i].plot(hardware_time_lf[:plot_len_lf], hardware_states[:plot_len_lf, 10+i+5], label="hardware")
        ax[1][i].set_title("Right " + titles[i])
        ax[1][i].set_xlabel("Time")
        ax[1][i].legend()

    fig.suptitle("Inputted Motor Pos")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if do_save:
        plt.savefig(os.path.join(save_dir, "motor_pos_in.png"))
    else:
        plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["X", "Y", "Z"]
    ax[0].set_ylabel("Pel Vel (m/s)")
    offset = 20
    for i in range(3):
        ax[i].plot(sim_time_lf[:plot_len_lf], sim_states[:plot_len_lf, offset+i], label="sim")
        ax[i].plot(hardware_time_lf[:plot_len_lf], hardware_states[:plot_len_lf, offset+i], label="hardware")
        ax[i].set_title(titles[i])
        ax[i].legend()
        ax[i].set_xlabel("Time")

    fig.suptitle("Inputted Pel Vel")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if do_save:
        plt.savefig(os.path.join(save_dir, "pel_vel_in.png"))
    else:
        plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Roll", "Pitch", "Yaw"]
    ax[0].set_ylabel("Pel Ang Vel (rad/s)")
    offset = 23
    for i in range(3):
        ax[i].plot(sim_time_lf[:plot_len_lf], sim_states[:plot_len_lf, offset+i], label="sim")
        ax[i].plot(hardware_time_lf[:plot_len_lf], hardware_states[:plot_len_lf, offset+i], label="hardware")
        ax[i].set_title(titles[i])
        ax[i].legend()
        ax[i].set_xlabel("Time")

    fig.suptitle("Inputted Pel Ang Vel")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if do_save:
        plt.savefig(os.path.join(save_dir, "pel_ang_vel_in.png"))
    else:
        plt.show()

    fig, ax = plt.subplots(2, 5, figsize=(15, 5))
    # t = np.linspace(0, num_steps-1, num_steps*simrate)
    titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
    ax[0][0].set_ylabel("Velocity (rad/s)")
    ax[1][0].set_ylabel("Velocity (rad/s)")
    offset = 26
    for i in range(5):
        
        ax[0][i].plot(sim_time_lf[:plot_len_lf], sim_states[:plot_len_lf, offset+i], label="sim")
        ax[0][i].plot(hardware_time_lf[:plot_len_lf], hardware_states[:plot_len_lf, offset+i], label="hardware")
        ax[0][i].set_title("Left " + titles[i])
        ax[0][i].legend()

        ax[1][i].plot(sim_time_lf[:plot_len_lf], sim_states[:plot_len_lf, offset+i+5], label="sim")
        ax[1][i].plot(hardware_time_lf[:plot_len_lf], hardware_states[:plot_len_lf, offset+i+5], label="hardware")
        ax[1][i].set_title("Right " + titles[i])
        ax[1][i].set_xlabel("Time")
        ax[1][i].legend()

    fig.suptitle("Inputted Motor Vel")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if do_save:
        plt.savefig(os.path.join(save_dir, "motor_vel_in.png"))
    else:
        plt.show()

    fig, ax = plt.subplots(2, 2, figsize=(15, 5))
    # t = np.linspace(0, num_steps-1, num_steps*simrate)
    titles = ["Shin", "Tarsus"]
    ax[0][0].set_ylabel("Pos (rad)")
    ax[1][0].set_ylabel("Pos (rad)")
    offset = 36
    for i in range(2):
        
        ax[0][i].plot(sim_time_lf[:plot_len_lf], sim_states[:plot_len_lf, offset+i], label="sim")
        ax[0][i].plot(hardware_time_lf[:plot_len_lf], hardware_states[:plot_len_lf, offset+i], label="hardware")
        ax[0][i].set_title("Left " + titles[i])
        ax[0][i].legend()

        ax[1][i].plot(sim_time_lf[:plot_len_lf], sim_states[:plot_len_lf, offset+i+2], label="sim")
        ax[1][i].plot(hardware_time_lf[:plot_len_lf], hardware_states[:plot_len_lf, offset+i+2], label="hardware")
        ax[1][i].set_title("Right " + titles[i])
        ax[1][i].set_xlabel("Time")
        ax[1][i].legend()

    fig.suptitle("Inputted Joint Pos")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if do_save:
        plt.savefig(os.path.join(save_dir, "joint_pos_in.png"))
    else:
        plt.show()

    fig, ax = plt.subplots(2, 2, figsize=(15, 5))
    titles = ["Shin", "Tarsus"]
    ax[0][0].set_ylabel("Vel (rad/s)")
    ax[1][0].set_ylabel("Vel (rad/s)")
    offset = 40
    for i in range(2):
        
        ax[0][i].plot(sim_time_lf[:plot_len_lf], sim_states[:plot_len_lf, offset+i], label="sim")
        ax[0][i].plot(hardware_time_lf[:plot_len_lf], hardware_states[:plot_len_lf, offset+i], label="hardware")
        ax[0][i].set_title("Left " + titles[i])
        ax[0][i].legend()

        ax[1][i].plot(sim_time_lf[:plot_len_lf], sim_states[:plot_len_lf, offset+i+2], label="sim")
        ax[1][i].plot(hardware_time_lf[:plot_len_lf], hardware_states[:plot_len_lf, offset+i+2], label="hardware")
        ax[1][i].set_title("Right " + titles[i])
        ax[1][i].set_xlabel("Time")
        ax[1][i].legend()

    fig.suptitle("Inputted Joint Vel")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if do_save:
        plt.savefig(os.path.join(save_dir, "joint_vel_in.png"))
    else:
        plt.show()

# Graph state data
if vis["mpos_input"]:
    fig, ax = plt.subplots(2, 5, figsize=(15, 5))
    # t = np.linspace(0, num_steps-1, num_steps*simrate)
    titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
    ax[0][0].set_ylabel("Position")
    ax[1][0].set_ylabel("Position")
    for i in range(5):
        if i == 1:
            ax[0][i].plot(sim_time_lf[:plot_len_lf], sim_states[:plot_len_lf, 10+i], label="sim")
            ax[0][i].plot(hardware_time_lf[:plot_len_lf], hardware_states[:plot_len_lf, 10+i], label="hardware")
            ax[0][i].set_title("Left " + titles[i])
            

            ax[1][i].plot(sim_time_lf[:plot_len_lf], sim_states[:plot_len_lf, 10+i+5], label="sim")
            ax[1][i].plot(hardware_time_lf[:plot_len_lf], hardware_states[:plot_len_lf, 10+i+5], label="hardware")
            ax[1][i].set_title("Right " + titles[i])
            ax[1][i].set_xlabel("Time")
            ax[0][i].legend()
            ax[1][i].legend()
        else:
            # ax[0][i].plot(t, torques[:, i])
            ax[0][i].plot(sim_time_lf[:plot_len_lf], sim_states[:plot_len_lf, 10+i], label="sim")
            ax[0][i].plot(hardware_time_lf[:plot_len_lf], hardware_states[:plot_len_lf, 10+i], label="hardware")
            ax[0][i].set_title("Left " + titles[i])
            ax[0][i].legend()

            ax[1][i].plot(sim_time_lf[:plot_len_lf], sim_states[:plot_len_lf, 10+i+5], label="sim")
            ax[1][i].plot(hardware_time_lf[:plot_len_lf], hardware_states[:plot_len_lf, 10+i+5], label="hardware")
            ax[1][i].set_title("Right " + titles[i])
            ax[1][i].set_xlabel("Time")
            ax[1][i].legend()

    fig.suptitle("Inputted Motor Pos")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if do_save:
        plt.savefig(os.path.join(save_dir, "motor_pos_in.png"))
    else:
        plt.show()

if vis["mvel_input"]:
    fig, ax = plt.subplots(2, 5, figsize=(15, 5))
    # t = np.linspace(0, num_steps-1, num_steps*simrate)
    titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
    ax[0][0].set_ylabel("Position")
    ax[1][0].set_ylabel("Position")
    for i in range(5):
       
        ax[0][i].plot(sim_time_lf[:plot_len_lf], sim_states[:plot_len_lf, 26+i], label="sim")
        ax[0][i].plot(hardware_time_lf[:plot_len_lf], hardware_states[:plot_len_lf, 26+i], label="hardware")
        ax[0][i].set_title("Left " + titles[i])
        ax[0][i].legend()

        ax[1][i].plot(sim_time_lf[:plot_len_lf], sim_states[:plot_len_lf, 26+i+5], label="sim")
        ax[1][i].plot(hardware_time_lf[:plot_len_lf], hardware_states[:plot_len_lf, 26+i+5], label="hardware")
        ax[1][i].set_title("Right " + titles[i])
        ax[1][i].set_xlabel("Time")
        ax[1][i].legend()

    fig.suptitle("Inputted Motor Vel")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if do_save:
        plt.savefig(os.path.join(save_dir, "motor_vel_in.png"))
    else:
        plt.show()

if vis["mpos"]:
    fig, ax = plt.subplots(2, 5, figsize=(15, 5))
    # t = np.linspace(0, num_steps-1, num_steps*simrate)
    titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
    ax[0][0].set_ylabel("Position")
    ax[1][0].set_ylabel("Position")
    for i in range(5):
       
        ax[0][i].plot(sim_time[:plot_len], sim_motor_pos[:plot_len, i], label="sim")
        ax[0][i].plot(hardware_time[:plot_len], hardware_motor_pos[:plot_len, i], label="hardware")
        ax[0][i].set_title("Left " + titles[i])
        ax[0][i].legend()

        ax[1][i].plot(sim_time[:plot_len], sim_motor_pos[:plot_len, i+5], label="sim")
        ax[1][i].plot(hardware_time[:plot_len], hardware_motor_pos[:plot_len, i+5], label="hardware")
        ax[1][i].set_title("Right " + titles[i])
        ax[1][i].set_xlabel("Time")
        ax[1][i].legend()

    fig.suptitle("Motor Pos")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if do_save:
        plt.savefig(os.path.join(save_dir, "motor_pos.png"))
    else:
        plt.show()


if vis["mvel"]:
    fig, ax = plt.subplots(2, 5, figsize=(15, 5))
    # t = np.linspace(0, num_steps-1, num_steps*simrate)
    titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
    ax[0][0].set_ylabel("Position")
    ax[1][0].set_ylabel("Position")
    for i in range(5):
       
        ax[0][i].plot(sim_time[:plot_len], sim_motor_vel[:plot_len, i], label="sim")
        ax[0][i].plot(hardware_time[:plot_len], hardware_motor_pos[:plot_len, i], label="hardware")
        ax[0][i].set_title("Left " + titles[i])
        ax[0][i].legend()

        ax[1][i].plot(sim_time[:plot_len], sim_motor_pos[:plot_len, i+5], label="sim")
        ax[1][i].plot(hardware_time[:plot_len], hardware_motor_pos[:plot_len, i+5], label="hardware")
        ax[1][i].set_title("Right " + titles[i])
        ax[1][i].set_xlabel("Time")
        ax[1][i].legend()

    fig.suptitle("Motor Vel")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if do_save:
        plt.savefig(os.path.join(save_dir, "motor_vel.png"))
    else:
        plt.show()

if vis["pel_vel"]:
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # t = np.linspace(0, num_steps-1, num_steps*simrate)
    titles = ["X", "Y", "Z"]
    ax[0].set_ylabel("Pel Vel (m/s)")
    offset = 20
    for i in range(3):
        # ax[0][i].plot(t, torques[:, i])
        ax[i].plot(sim_time_lf[:plot_len_lf], sim_states[:plot_len_lf, offset+i], label="sim")
        ax[i].plot(hardware_time_lf[:plot_len_lf], hardware_states[:plot_len_lf, offset+i], label="hardware")
        # ax[0][i].plot(sim_torques[:plot_len, i], label="sim")
        # ax[0][i].plot(hardware_torques[:plot_len, i], label="hardware")
        ax[i].set_title(titles[i])
        ax[i].legend()
        # ax[1][i].plot(t, torques[:, i+5])
        ax[i].set_xlabel("Time")

    fig.suptitle("Inputted Pel Vel")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if do_save:
        plt.savefig(os.path.join(save_dir, "pel_vel_in.png"))
    else:
        plt.show()

if vis["footpos"]:
    fig, ax = plt.subplots(2, 3, figsize=(15, 5))
    # t = np.linspace(0, num_steps-1, num_steps*simrate)
    titles = ["X", "Y", "Z"]
    ax[0][0].set_ylabel("Foot Pos")
    ax[1][0].set_ylabel("Foot Pos")

    offset = 0
    for i in range(3):
        # ax[0][i].plot(t, torques[:, i])
        ax[0][i].plot(sim_time_lf[:plot_len_lf], sim_states[:plot_len_lf, offset+i], label="sim")
        ax[0][i].plot(hardware_time_lf[:plot_len_lf], hardware_states[:plot_len_lf, offset+i], label="hardware")
        ax[0][i].set_title("Left " + titles[i])
        ax[0][i].legend()

        ax[1][i].plot(sim_time_lf[:plot_len_lf], sim_states[:plot_len_lf, offset+i+3], label="sim")
        ax[1][i].plot(hardware_time_lf[:plot_len_lf], hardware_states[:plot_len_lf, offset+i+3], label="hardware")
        ax[1][i].set_title("Right " + titles[i])
        ax[1][i].legend()
        # ax[0][i].plot(sim_torques[:plot_len, i], label="sim")
        # ax[0][i].plot(hardware_torques[:plot_len, i], label="hardware")
        # ax[1][i].plot(t, torques[:, i+5])
        ax[1][i].set_xlabel("Time")

    fig.suptitle("Inputted Foot Pos")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if do_save:
        plt.savefig(os.path.join(save_dir, "foot_pos_in.png"))
    else:
        plt.show()

if vis["torques"]:
    # Graph torque data
    fig, ax = plt.subplots(2, 5, figsize=(15, 5))
    # t = np.linspace(0, num_steps-1, num_steps*simrate)
    titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
    ax[0][0].set_ylabel("Torque")
    ax[1][0].set_ylabel("Torque")
    for i in range(5):
        # ax[0][i].plot(t, torques[:, i])
        ax[0][i].plot(sim_time[:plot_len], sim_torques[:plot_len, i], label="sim")
        ax[0][i].plot(hardware_time[:plot_len], hardware_torques[:plot_len, i], label="hardware")
        # ax[0][i].plot(sim_torques[:plot_len, i], label="sim")
        # ax[0][i].plot(hardware_torques[:plot_len, i], label="hardware")
        ax[0][i].set_title("Left " + titles[i])
        ax[0][i].legend()
        # ax[1][i].plot(t, torques[:, i+5])
        ax[1][i].plot(sim_time[:plot_len], sim_torques[:plot_len, i+5], label="sim")
        ax[1][i].plot(hardware_time[:plot_len], hardware_torques[:plot_len, i+5], label="hardware")
        # ax[1][i].plot(sim_torques[:plot_len, i+5], label="sim")
        # ax[1][i].plot(hardware_torques[:plot_len, i+5], label="hardware")
        ax[1][i].set_title("Right " + titles[i])
        ax[1][i].set_xlabel("Timesteps (0.03 sec)")
        ax[1][i].legend()

    fig.suptitle("Time Series Torque")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if do_save:
        plt.savefig(os.path.join(save_dir, "torque.png"))
    else:
        plt.show()

if vis["mpos_diff"]:
    fig, ax = plt.subplots(2, 5, figsize=(15, 5))
    # plot_len -= 1
    titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
    ax[0][0].set_ylabel("Motor Pos")
    ax[1][0].set_ylabel("Motor Pos")
    for i in range(5):
        ax[0][i].plot(sim_time[:plot_len-1], sim_motors_diff[:plot_len-1, i], label="sim")
        ax[0][i].plot(hardware_time[:plot_len-1], hardware_motors_diff[:plot_len-1, i], label="hardware")

        ax[0][i].set_title("Left " + titles[i])
        ax[0][i].legend()
        ax[1][i].plot(sim_time[:plot_len-1], sim_motors_diff[:plot_len-1, i+5], label="sim")
        ax[1][i].plot(hardware_time[:plot_len-1], hardware_motors_diff[:plot_len-1, i+5], label="hardware")

        ax[1][i].set_title("Right " + titles[i])
        ax[1][i].set_xlabel("Timesteps (0.03 sec)")
        ax[1][i].legend()

    fig.suptitle("Change in Motor Pos")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if do_save:
        plt.savefig(os.path.join(save_dir, "motor_pos_diff.png"))
    else:
        plt.show()


if vis["torque_mpos_diff"]:
    fig, ax = plt.subplots(2, 5, figsize=(15, 5))
    # plot_len -= 1
    # plot_len = plot_len // 4
    # t = np.linspace(0, num_steps-1, num_steps*simrate)
    # print("torques:", sim_torques.shape)
    # print("motors:", sim_motors.shape)
    titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
    ax[0][0].set_ylabel("Motor Pos")
    ax[1][0].set_ylabel("Motor Pos")
    for i in range(5):
        # ax[0][i].plot(t, torques[:, i])
        
        ax[0][i].scatter(sim_torques[:plot_len-1, i], sim_motors_diff[:plot_len-1, i], label="sim", alpha=0.7, c='C0')
        ax[0][i].scatter(hardware_torques[:plot_len-1, i], hardware_motors_diff[:plot_len-1, i], label="hardware", alpha=0.7, c='C1')
        
        # ax[0][i].plot(sim_torques[:plot_len, i], label="sim")
        # ax[0][i].plot(hardware_torques[:plot_len, i], label="hardware")
        ax[0][i].set_title("Left " + titles[i])
        ax[0][i].legend()
        # ax[1][i].plot(t, torques[:, i+5])
        
        ax[1][i].scatter(sim_torques[:plot_len-1, i+5], sim_motors_diff[:plot_len-1, i+5], label="sim", alpha=0.7, c='C0')
        ax[1][i].scatter(hardware_torques[:plot_len-1, i+5], hardware_motors_diff[:plot_len-1, i+5], label="hardware", alpha=0.7, c='C1')
        
        # ax[1][i].plot(sim_torques[:plot_len, i+5], label="sim")
        # ax[1][i].plot(hardware_torques[:plot_len, i+5], label="hardware")
        ax[1][i].set_title("Right " + titles[i])
        ax[1][i].set_xlabel("Torque (N/m)")
        ax[1][i].legend()

    fig.suptitle("Torque vs. Change in Motor Pos")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if do_save:
        plt.savefig(os.path.join(save_dir, "torque_motor_diff.png"))
    else:
        plt.show()

if vis["torque_pos_error"]:
    fig, ax = plt.subplots(2, 5, figsize=(15, 5))
    # plot_len -= 1
    # plot_len = plot_len // 4
    # t = np.linspace(0, num_steps-1, num_steps*simrate)
    titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
    ax[0][0].set_ylabel("Torque")
    ax[1][0].set_ylabel("Torque")
    for i in range(5):
        # ax[0][i].plot(t, torques[:, i])
        ax[0][i].scatter(sim_targets[:orig_plot_len-1, i] - sim_motor_pos[:orig_plot_len-1, i], sim_torques[1:orig_plot_len, i], label="sim NN", alpha=0.7, s=0.1)
        ax[0][i].scatter(hardware_targets[:orig_plot_len-1, i] - hardware_motor_pos[:orig_plot_len-1, i], hardware_torques[1:orig_plot_len, i], label="hardware", alpha=0.7, s=0.1)
        # ax[0][i].plot(sim_torques[:plot_len, i], label="sim")
        # ax[0][i].plot(hardware_torques[:plot_len, i], label="hardware")
        ax[0][i].set_title("Left " + titles[i])
        ax[0][i].legend()
        # ax[1][i].plot(t, torques[:, i+5])
        ax[1][i].scatter(sim_targets[:orig_plot_len-1, i+5] - sim_motor_pos[:orig_plot_len-1, i+5], sim_torques[1:orig_plot_len, i+5], label="sim NN", alpha=0.7, s=0.1)
        ax[1][i].scatter(hardware_targets[:orig_plot_len-1, i+5] - hardware_motor_pos[:orig_plot_len-1, i+5], hardware_torques[1:orig_plot_len, i+5], label="hardware", alpha=0.7, s=0.1)
        # ax[1][i].plot(sim_torques[:plot_len, i+5], label="sim")
        # ax[1][i].plot(hardware_torques[:plot_len, i+5], label="hardware")
        ax[1][i].set_title("Right " + titles[i])
        ax[1][i].set_xlabel("Position Error")
        ax[1][i].legend()

    fig.suptitle("Torque vs. Position Error")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if do_save:
        plt.savefig(os.path.join(save_dir, "torque_pos_err.png"))
    else:
        plt.show()

if vis["torque_pos_error_knee"]:
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    # plot_len -= 1
    # plot_len = plot_len // 4
    # t = np.linspace(0, num_steps-1, num_steps*simrate)
    title = "Knee"
    ind = 3
    ax[0].set_ylabel("Torque")

    ax[0].scatter(sim_targets[:orig_plot_len-1, ind] - sim_motor_pos[:orig_plot_len-1, ind], sim_torques[1:orig_plot_len, ind], label="sim NN", alpha=0.7, s=0.1)
    ax[0].scatter(hardware_targets[:orig_plot_len-1, ind] - hardware_motor_pos[:orig_plot_len-1, ind], hardware_torques[1:orig_plot_len, ind], label="hardware", alpha=0.7, s=0.1)
    ax[0].set_title("Left " + title)
    ax[1].set_xlabel("Position Error")
    ax[0].legend()

    ax[1].scatter(sim_targets[:orig_plot_len-1, ind+5] - sim_motor_pos[:orig_plot_len-1, ind+5], sim_torques[1:orig_plot_len, ind+5], label="sim NN", alpha=0.7, s=0.1)
    ax[1].scatter(hardware_targets[:orig_plot_len-1, ind+5] - hardware_motor_pos[:orig_plot_len-1, ind+5], hardware_torques[1:orig_plot_len, ind+5], label="hardware", alpha=0.7, s=0.1)
    ax[1].set_title("Right " + title)
    ax[1].set_xlabel("Position Error")
    ax[1].legend()

    fig.suptitle("Knee Torque vs. Position Error")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if do_save:
        plt.savefig(os.path.join(save_dir, "torque_pos_err_knee.png"))
    else:
        plt.show()

if vis["targets"]:
    fig, ax = plt.subplots(2, 5, figsize=(15, 5))
    # plot_len -= 1
    # plot_len = plot_len // 4
    # t = np.linspace(0, num_steps-1, num_steps*simrate)
    titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
    ax[0][0].set_ylabel("Target Motor Pos")
    ax[1][0].set_ylabel("Target Motor Pos")
    for i in range(5):
        # ax[0][i].plot(t, torques[:, i])
        ax[0][i].plot(sim_time[:orig_plot_len], sim_targets[:orig_plot_len, i], label="sim NN", alpha=0.7)
        ax[0][i].plot(hardware_time[:orig_plot_len], hardware_targets[:orig_plot_len, i], label="hardware", alpha=0.7)
        # ax[0][i].plot(sim_torques[:plot_len, i], label="sim")
        # ax[0][i].plot(hardware_torques[:plot_len, i], label="hardware")
        ax[0][i].set_title("Left " + titles[i])
        ax[0][i].legend()
        # ax[1][i].plot(t, torques[:, i+5])
        ax[1][i].plot(sim_time[:orig_plot_len], sim_targets[:orig_plot_len, i+5], label="sim", alpha=0.7)
        ax[1][i].plot(hardware_time[:orig_plot_len], hardware_targets[:orig_plot_len, i+5], label="hardware", alpha=0.7)
        # ax[1][i].plot(sim_torques[:plot_len, i+5], label="sim")
        # ax[1][i].plot(hardware_torques[:plot_len, i+5], label="hardware")
        ax[1][i].set_title("Right " + titles[i])
        ax[1][i].set_xlabel("Time")
        ax[1][i].legend()

    fig.suptitle("Commanded PD Targets")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if do_save:
        plt.savefig(os.path.join(save_dir, "pd_targ.png"))
    else:
        plt.show()