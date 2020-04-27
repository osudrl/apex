import sys, os
sys.path.append("..") # Adds higher directory to python modules path.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import torch
import time
import cmath
import math
import ray
from functools import partial

# from cassie import CassieEnv

def quaternion2euler(quaternion):
	w = quaternion[0]
	x = quaternion[1]
	y = quaternion[2]
	z = quaternion[3]
	ysqr = y * y
	
	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + ysqr)
	X = math.degrees(math.atan2(t0, t1))
	
	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	Y = math.degrees(math.asin(t2))
	
	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (ysqr + z * z)
	Z = math.degrees(math.atan2(t3, t4))

	result = np.zeros(3)
	result[0] = X * np.pi / 180
	result[1] = Y * np.pi / 180
	result[2] = Z * np.pi / 180
	
	return result

@torch.no_grad()
def eval_mission(cassie_env, policy, num_iters=2):
    # save data holds deviation between robot xy pos, z orient, xy velocity and specified pos, orient, velocity from mission
    # if mission ends early (robot height fall over indicator) 
    
    runs = []
    pass_data = np.zeros(num_iters) # whether or not robot stayed alive during mission

    for j in range(num_iters):
        mission_len = cassie_env.command_traj.trajlen
        run_data = []
        state = torch.Tensor(cassie_env.reset_for_test())
        count, passed, done = 0, 1, False
        while count < mission_len and not done:
            # cassie_env.render()
            # Get action and act
            action = policy(state, True)
            action = action.data.numpy()
            state, reward, done, _ = cassie_env.step(action)
            state = torch.Tensor(state)
            # See if end state reached
            if done or cassie_env.sim.qpos()[2] < 0.4:
                passed = 0
                print("mission failed")
            # Get command info, robot info
            commanded_pos = cassie_env.command_traj.global_pos[:,0:2]
            commanded_speed = cassie_env.command_traj.speed_cmd
            commanded_orient = cassie_env.command_traj.orient
            qpos = cassie_env.sim.qpos()
            qvel = cassie_env.sim.qvel()
            actual_pos = qpos[0:2] # only care about x and y
            actual_speed = np.linalg.norm(qvel[0:2])
            actual_orient = quaternion2euler(qpos[3:7])[2] # only care about yaw
            # Calculate pos,vel,orient deviation as vector difference
            pos_error = np.linalg.norm(actual_pos - commanded_pos)
            speed_error = np.linalg.norm(actual_speed - commanded_speed)
            orient_error = np.linalg.norm(actual_orient - commanded_orient)
            # Log info
            run_data.append(([count, pos_error, speed_error, orient_error]))
            count += 1
        if passed:
            print("mission passed")
            pass_data[j] = 1
        runs.append(np.array(run_data))

    # summary stats
    run_lens = [len(run) for run in runs]
    print("longest / shortest / average steps : {} / {} / {}".format(max(run_lens), min(run_lens), sum(run_lens) / len(run_lens)))

    save_data = dict()
    save_data["runs"] = runs
    save_data["pass"] = pass_data

    return save_data


def plot_mission_data(save_data, missions):
    num_missions = len(save_data)
    fig, axs = plt.subplots(num_missions, 3, figsize=(num_missions*5, 15))
    for i in range(num_missions):
        mission_runs = save_data[i]["runs"]
        for run in mission_runs:
            axs[i][0].plot(run[:, 0], run[:, 1])
            axs[i][1].plot(run[:, 0], run[:, 2])
            axs[i][2].plot(run[:, 0], run[:, 3])
        axs[i][1].set_title(missions[i]) # only put title on middle plot
        [axs[i][j].set_xlabel("steps") for j in range(3)]
        [axs[i][j].set_ylabel("error") for j in range(3)]
    plt.tight_layout(pad=3.0)
    plt.show()