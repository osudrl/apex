import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib as mpl
import torch
from torch.autograd import Variable
import time
import cmath

from cassie import CassieEnv
from cassie.speed_no_delta_neutral_foot_env import CassieEnv_speed_no_delta_neutral_foot
from cassie.speed_sidestep_env import CassieEnv_speed_sidestep

from rl.policies import GaussianMLP

def compute_perturbs(cassie_env, policy, wait_time, perturb_duration, perturb_size, perturb_incr, perturb_body, num_angles):
    perturb_dir = -2*np.pi*np.linspace(0, 1, num_angles+1)  # Angles from straight forward to apply force

    # Get states at each phase:
    num_steps = cassie_env.phaselen + 1
    state = torch.Tensor(cassie_env.reset_for_test())
    cassie_env.speed = 0.5
    cassie_env.side_speed = 0
    cassie_env.phase_add = 1
    # Simulate for "wait_time" first to stabilize
    for i in range(num_steps*4):
        _, action = policy.act(state, True)
        action = action.data.numpy()
        state, reward, done, _ = cassie_env.step(action)
        state = torch.Tensor(state)
    qpos_phase = np.zeros((35, num_steps))
    qvel_phase = np.zeros((32, num_steps))
    # print("phase: ", cassie_env.phase)
    qpos_phase[:, 0] = cassie_env.sim.qpos()
    qvel_phase[:, 0] = cassie_env.sim.qvel()
    for i in range(num_steps-1):
        _, action = policy.act(state, True)
        action = action.data.numpy()
        state, reward, done, _ = cassie_env.step(action)
        state = torch.Tensor(state)
        # print("phase: ", cassie_env.phase)
        qpos_phase[:, i+1] = cassie_env.sim.qpos()
        qvel_phase[:, i+1] = cassie_env.sim.qvel()
    max_force = np.zeros((num_steps, num_angles))

    eval_start = time.time()
    for i in range(num_angles):
        for j in range(num_steps):
            print("Testing angle {} ({} out of {}) for phase {}".format(-perturb_dir[i], i+1, num_angles, j))
            curr_size = perturb_size - perturb_incr
            done = False
            while not done:
                curr_size += perturb_incr
                state = torch.Tensor(cassie_env.reset_for_test())
                cassie_env.speed = 0.5
                cassie_env.side_speed = 0
                cassie_env.phase_add = 1
                cassie_env.sim.set_qpos(qpos_phase[:, j])
                cassie_env.sim.set_qvel(qvel_phase[:, j])
                cassie_env.phase = j
                curr_time = 0
                # # Simulate for "wait_time" first to stabilize
                # while curr_time < wait_time:
                #     _, action = policy.act(state, True)
                #     action = action.data.numpy()
                #     state, reward, done, _ = cassie_env.step(action)
                #     state = torch.Tensor(state)
                #     curr_time = cassie_env.sim.time()
                # Apply perturb
                force_x = curr_size * np.cos(perturb_dir[i])
                force_y = curr_size * np.sin(perturb_dir[i])
                start_t = curr_time
                while curr_time < start_t + perturb_duration:
                    cassie_env.sim.apply_force([force_x, force_y, 0, 0, 0, 0], perturb_body)
                    _, action = policy.act(state, True)
                    action = action.data.numpy()
                    state, reward, done, _ = cassie_env.step(action)
                    state = torch.Tensor(state)
                    curr_time = cassie_env.sim.time()
                # Done perturbing, reset perturb_time and xfrc_applied
                cassie_env.sim.apply_force([0, 0, 0, 0, 0, 0], perturb_body)
                start_t = curr_time
                while curr_time < start_t + wait_time:
                    _, action = policy.act(state, True)
                    action = action.data.numpy()
                    state, reward, done, _ = cassie_env.step(action)
                    state = torch.Tensor(state)
                    curr_time = cassie_env.sim.time()
                    if cassie_env.sim.qpos()[2] < 0.4:  # Failed, reset and record force
                        done = True
                        break
            max_force[j, i] = curr_size - perturb_incr
            print("max force: ", curr_size - perturb_incr)

    print("Total compute time: ", time.time() - eval_start)
    np.save("test_perturb_eval_phase.npy", max_force)

def plot_perturb(filename):
    data = np.load(filename)
    data = np.mean(data, axis=0)
    print("data: ", data.shape)
    num_angles = len(data)
    angles = -2*np.pi*np.linspace(0, 1, num_angles+1)
    max_force = 50*np.ceil(np.max(data) / 50)
    print("max force: ", max_force)

    # fig = plt.figure()
    # clist = [(0, "red"), (0.125, "red"), (0.25, "orange"), (0.5, "green"), 
    #      (0.7, "green"), (0.75, "blue"), (1, "blue")]
    # rvb = mcolors.LinearSegmentedColormap.from_list("", clist)
    # # norm = mpl.colors.Normalize(0, max_force)
    # # cmap = cm.get_cmap("plasma")
    # plt.polar(angles[:-1], data, color=rvb(data/max_force))
    # ax = fig.gca()
    # ax.set_ylim([0,max_force])
    # plt.show()

    num_cells = 100
    fig, ax1 = plt.subplots(subplot_kw=dict(projection='polar'))
    ax1.patch.set_alpha(0)
    offset = 2*np.pi/100/2
    theta, r = np.mgrid[0-offset:2*np.pi-offset:complex(0,num_angles), 0:max_force:complex(0, num_cells)]
    print(theta.shape)
    color_data = np.zeros(theta.shape)
    for i in range(color_data.shape[0]):
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
    img = plt.imread("./cassie_top_white.png")
    ax_image.imshow(img, alpha=.3)
    ax_image.axis('off')
    # plt.show()
    plt.savefig("./test_perturb_eval_phase.png")

plot_perturb("./test_perturb_eval_phase.npy")
exit()

# Load environment and policy
cassie_env = CassieEnv_speed_no_delta_neutral_foot("walking", clock_based=True, state_est=True)
# cassie_env = CassieEnv_speed_sidestep("walking", simrate = 15, clock_based=True, state_est=True)

# file_prefix = "fwrd_walk_StateEst_speed-05-3_freq1-2_footvelpenalty_heightflag_footxypenalty"
# file_prefix = "sidestep_StateEst_speedmatch_footytraj_doublestance_time0.4_land0.2_vels_avgdiff_simrate15_evenweight_actpenalty"
file_prefix = "nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2"
policy = torch.load("./trained_models/{}.pt".format(file_prefix))
policy.bounded = False
policy.eval()

wait_time = 4
perturb_duration = 0.2
perturb_size = 100
perturb_incr = 10
perturb_body = "cassie-pelvis"
num_angles = 100

compute_perturbs(cassie_env, policy, wait_time, perturb_duration, perturb_size, perturb_incr, perturb_body, num_angles)
