import sys, os
sys.path.append("..") # Adds higher directory to python modules path.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import torch
import time
import cmath
import ray
from functools import partial

# from cassie import CassieEnv

# from rl.policies.actor import GaussianMLP_Actor

# Will reset the env to the given phase by reset_for_test, and then
# simulating 2 cycle then to the given phase
@torch.no_grad()
def reset_to_phase(env, policy, phase):
    state = torch.Tensor(env.reset_for_test(full_reset=True))
    env.speed = 0.5
    for i in range(2*(env.phaselen + 1)):
        action = policy(state, True)
        action = action.data.numpy()
        state, reward, done, _ = env.step(action)
        state = torch.Tensor(state)
    for i in range(phase):
        action = policy(state, True)
        action = action.data.numpy()
        state, reward, done, _ = env.step(action)
        state = torch.Tensor(state)
    return state

@torch.no_grad()
def compute_perturbs(cassie_env, policy, wait_time=4, perturb_duration=0.2, perturb_size=100, perturb_incr=10, perturb_body="cassie-pelvis", num_angles="4"):
    perturb_dir = -2*np.pi*np.linspace(0, 1, num_angles+1)  # Angles from straight forward to apply force

    # Get states at each phase:
    num_steps = cassie_env.phaselen + 1
    state = torch.Tensor(cassie_env.reset_for_test(full_reset=True))
    max_force = np.zeros((num_steps, num_angles))

    eval_start = time.time()
    for i in range(num_angles):
        for j in range(num_steps):
            print("Testing angle {} ({} out of {}) for phase {}".format(-perturb_dir[i], i+1, num_angles, j))
            reset_to_phase(cassie_env, policy, j)
            curr_size = perturb_size - perturb_incr
            done = False
            curr_start = time.time()
            while not done:
                curr_size += perturb_incr
                print("curr size: ", curr_size)
                reset_to_phase(cassie_env, policy, j)
                curr_time = cassie_env.sim.time()
                # Apply perturb
                force_x = curr_size * np.cos(perturb_dir[i])
                force_y = curr_size * np.sin(perturb_dir[i])
                start_t = curr_time
                while curr_time < start_t + perturb_duration:
                    cassie_env.sim.apply_force([force_x, force_y, 0, 0, 0, 0], perturb_body)
                    action = policy(state, True)
                    action = action.data.numpy()
                    state, reward, done, _ = cassie_env.step(action)
                    state = torch.Tensor(state)
                    curr_time = cassie_env.sim.time()
                # Done perturbing, reset perturb_time and xfrc_applied
                cassie_env.sim.apply_force([0, 0, 0, 0, 0, 0], perturb_body)
                start_t = curr_time
                while curr_time < start_t + wait_time:
                    action = policy(state, True)
                    action = action.data.numpy()
                    state, reward, done, _ = cassie_env.step(action)
                    state = torch.Tensor(state)
                    curr_time = cassie_env.sim.time()
                    if cassie_env.sim.qpos()[2] < 0.4:  # Failed, reset and record force
                        done = True
                        break
            max_force[j, i] = curr_size - perturb_incr
            print("max force: ", curr_size - perturb_incr)
            print("search time: ", time.time() - curr_start)

    print("Total compute time: ", time.time() - eval_start)
    return max_force
    # np.save("test_perturb_eval_phase.npy", max_force)

@ray.remote
@torch.no_grad()
def perturb_worker(env_fn, qpos_phase, qvel_phase, policy, angles, wait_time, perturb_duration, perturb_size, perturb_incr, perturb_body, worker_id):
    num_steps = qpos_phase.shape[1]
    num_angles = len(angles)
    max_force = np.zeros((num_steps, num_angles))
    cassie_env = env_fn()

    eval_start = time.time()
    sim_times = np.zeros((num_angles, num_steps))
    for i in range(num_angles):
        for j in range(num_steps):
            sim_start = time.time()
            # print("Testing angle {} ({} out of {}) for phase {}".format(-angles[i], i+1, num_angles, j))
            curr_size = perturb_size - perturb_incr
            done = False
            while not done:
                curr_size += perturb_incr
                # state = torch.Tensor(cassie_env.full_reset())
                # cassie_env.speed = 0.5
                # cassie_env.side_speed = 0
                # cassie_env.phase_add = 1
                # cassie_env.sim.set_qpos(qpos_phase[:, j])
                # cassie_env.sim.set_qvel(qvel_phase[:, j])
                # cassie_env.phase = j
                state = reset_to_phase(cassie_env, policy, j)
                curr_time = 0
                # Apply perturb
                force_x = curr_size * np.cos(angles[i])
                force_y = curr_size * np.sin(angles[i])
                start_t = curr_time
                while curr_time < start_t + perturb_duration:
                    cassie_env.sim.apply_force([force_x, force_y, 0, 0, 0, 0], perturb_body)
                    action = policy(state, True)
                    action = action.data.numpy()
                    state, reward, done, _ = cassie_env.step(action)
                    state = torch.Tensor(state)
                    curr_time = cassie_env.sim.time()
                # Done perturbing, reset xfrc_applied
                cassie_env.sim.apply_force([0, 0, 0, 0, 0, 0], perturb_body)
                start_t = curr_time
                while curr_time < start_t + wait_time:
                    action = policy(state, True)
                    action = action.data.numpy()
                    state, reward, done, _ = cassie_env.step(action)
                    state = torch.Tensor(state)
                    curr_time = cassie_env.sim.time()
                    if cassie_env.sim.qpos()[2] < 0.4:  # Failed, reset and record force
                        done = True
                        break
            max_force[j, i] = curr_size - perturb_incr
            sim_times[i, j] = time.time() - sim_start
            # print("max force: ", curr_size - perturb_incr)
    return max_force, time.time() - eval_start, sim_times, worker_id
    # print("Total compute time: ", time.time() - eval_start)

def compute_perturbs_multi(env_fn, policy, wait_time=4, perturb_duration=0.2, perturb_size=100, perturb_incr=10, perturb_body="cassie-pelvis", num_angles=4, num_procs=4):
    perturb_dir = -2*np.pi*np.linspace(0, 1, num_angles+1)  # Angles from straight forward to apply force
    cassie_env = env_fn()
    # Get states at each phase:
    num_steps = cassie_env.phaselen + 1
    state = torch.Tensor(cassie_env.reset_for_test(full_reset=True))
    cassie_env.speed = 0.5
    cassie_env.side_speed = 0
    cassie_env.phase_add = 1
    # Simulate for 4 cycles first to stabilize
    for i in range(num_steps*4):
        action = policy(state, True)
        action = action.data.numpy()
        state, reward, done, _ = cassie_env.step(action)
        state = torch.Tensor(state)
    qpos_phase = np.zeros((35, num_steps))
    qvel_phase = np.zeros((32, num_steps))
    qpos_phase[:, 0] = cassie_env.sim.qpos()
    qvel_phase[:, 0] = cassie_env.sim.qvel()
    for i in range(num_steps-1):
        action = policy(state, True)
        action = action.data.numpy()
        state, reward, done, _ = cassie_env.step(action)
        state = torch.Tensor(state)
        # print("phase: ", cassie_env.phase)
        qpos_phase[:, i+1] = cassie_env.sim.qpos()
        qvel_phase[:, i+1] = cassie_env.sim.qvel()

    start_t = time.time()
    ray.init(num_cpus=num_procs)
    result_ids = []
    if num_procs > num_angles:
        num_procs = num_angles
    angle_split = (num_angles) // num_procs
    for i in range(num_procs):
        print("Start ind: {} End ind: {}".format(angle_split*i, angle_split*(i+1)))
        args = (env_fn, qpos_phase, qvel_phase, policy, perturb_dir[angle_split*i:angle_split*(i+1)], wait_time, perturb_duration, 
                    perturb_size, perturb_incr, perturb_body, i)
        print("Starting worker ", i)
        result_ids.append(perturb_worker.remote(*args))
    result = ray.get(result_ids)
    print(result)
    print("Got all results")
    max_force = np.concatenate([result[i][0] for i in range(num_procs)], axis=1)
    print("timings: ", [result[i][1] for i in range(num_procs)])
    print("sim timings: ", [result[i][2] for i in range(num_procs)])
    # max_force = np.concatenate(result, axis=1)
    # print("max force: ", max_force)
    print("total time: ", time.time() - start_t)
    ray.shutdown()
    return max_force


def plot_perturb(filename):
    data = np.load(filename)
    print("max data: ", np.max(data))
    data = np.mean(data, axis=0)
    print("data: ", data.shape)
    num_angles = len(data)
    
    max_force = 50*np.ceil(np.max(data) / 50)
    print("max force: ", max_force)

    num_cells = 100
    fig, ax1 = plt.subplots(subplot_kw=dict(projection='polar'))
    ax1.patch.set_alpha(0)
    offset = 2*np.pi/100/2
    theta, r = np.mgrid[0-offset:2*np.pi-offset:complex(0,num_angles+1), 0:max_force:complex(0, num_cells)]
    print(theta.shape)
    color_data = np.zeros(theta.shape)
    for i in range(color_data.shape[0]-1):
        idx = int(np.floor(data[i] / (max_force / num_cells)))
        curr_data = data[i]
        color_data[i][idx] = data[i]
        for j in range(idx+1, num_cells):
            color_data[i][j] = 0
        for j in range(1, idx+1):
            color_data[i][idx-j] = curr_data - max_force / num_cells * j
        # color_data[i, :] = np.linspace(0, max_force, color_data.shape[1])
    norm = mpl.colors.Normalize(0.0, max_force)
    # clist = [(0, "white"), (1, "green")]
    clist = [(0, "white"), (0.25, "red"), (0.5, "orange"), (0.75, "yellow"), (1, "green")]
    rvb = mcolors.LinearSegmentedColormap.from_list("", clist, N=200)
    pcm = ax1.pcolormesh(theta, r, color_data, cmap=rvb, norm=norm)
    ax1.set_ylim([0, max_force])
    
    cbaxes = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    plt.colorbar(pcm, cax=cbaxes)
    ax1.grid()
    ax_image = fig.add_axes([-0.03, 0, 1, 1])
    img = plt.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cassie_top_white.png"))
    ax_image.imshow(img, alpha=.3)
    ax_image.axis('off')
    plt.show()
    plt.savefig("./test_perturb_eval_phase.png")

# plot_perturb("./test_perturb_eval_phase.npy")
# exit()

# import argparse
# import pickle

# parser = argparse.ArgumentParser()
# parser.add_argument("--path", type=str, default=None, help="path to folder containing policy and run details")
# args = parser.parse_args()
# run_args = pickle.load(open(args.path + "experiment.pkl", "rb"))

# # RUN_NAME = "7b7e24-seed0"
# # POLICY_PATH = "../trained_models/ppo/Cassie-v0/" + RUN_NAME + "/actor.pt"

# # Load environment and policy
# # env_fn = partial(CassieEnv_speed_no_delta_neutral_foot, "walking", clock_based=True, state_est=True)
# cassie_env = CassieEnv(traj=run_args.traj, clock_based=run_args.clock_based, state_est=run_args.state_est, dynamics_randomization=run_args.dyn_random)
# policy = torch.load(args.path + "actor.pt")

# wait_time = 4
# perturb_duration = 0.2
# perturb_size = 100
# perturb_incr = 10
# perturb_body = "cassie-pelvis"
# num_angles = 4
# num_procs = num_angles

# compute_perturbs(cassie_env, policy, wait_time, perturb_duration, perturb_size, perturb_incr, perturb_body, num_angles)
# compute_perturbs_multi(env_fn, policy, wait_time, perturb_duration, perturb_size, perturb_incr, perturb_body, num_angles, num_procs)
