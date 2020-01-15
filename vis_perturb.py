import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import time

from rl.utils import renderpolicy
from cassie import CassieEnv
from cassie.no_delta_env import CassieEnv_nodelta
from cassie.speed_env import CassieEnv_speed
from cassie.speed_double_freq_env import CassieEnv_speed_dfreq
from cassie.speed_no_delta_neutral_foot_env import CassieEnv_speed_no_delta_neutral_foot
from cassie.speed_sidestep_env import CassieEnv_speed_sidestep

from rl.policies import GaussianMLP

# Load environment and policy
# cassie_env = CassieEnv("walking", clock_based=True, state_est=False)
# cassie_env = CassieEnv_nodelta("walking", clock_based=True, state_est=False)
# cassie_env = CassieEnv_speed("walking", clock_based=True, state_est=True)
# cassie_env = CassieEnv_speed_dfreq("walking", clock_based=True, state_est=False)
cassie_env = CassieEnv_speed_no_delta_neutral_foot("walking", clock_based=True, state_est=True)
# cassie_env = CassieEnv_speed_sidestep("walking", simrate = 15, clock_based=True, state_est=True)

obs_dim = cassie_env.observation_space.shape[0] # TODO: could make obs and ac space static properties
action_dim = cassie_env.action_space.shape[0]

no_delta = True
offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])

# file_prefix = "fwrd_walk_StateEst_speed-05-3_freq1-2_footvelpenalty_heightflag_footxypenalty"
# file_prefix = "sidestep_StateEst_speedmatch_footytraj_doublestance_time0.4_land0.2_vels_avgdiff_simrate15_evenweight_actpenalty"
file_prefix = "nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2"
policy = torch.load("./trained_models/{}.pt".format(file_prefix))
policy.bounded = False
policy.eval()

state = torch.Tensor(cassie_env.reset_for_test())
cassie_env.speed = 0.5
cassie_env.side_speed = 0
cassie_env.phase_add = 1
num_steps = cassie_env.phaselen + 1
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

state = torch.Tensor(cassie_env.reset_for_test())

cassie_env.speed = 0.5
cassie_env.side_speed = 0
cassie_env.phase_add = 1
wait_time = 4
dt = 0.05
speedup = 3
perturb_time = 0
perturb_duration = 0.2
perturb_size = 150
perturb_dir = -2*np.pi*np.linspace(0, 1, 5)  # Angles from straight forward to apply force
perturb_body = "cassie-pelvis"
dir_idx = 0

###### Vis a single Perturbation for a given phase ######
# test_phase = 3
# cassie_env.sim.set_qpos(qpos_phase[:, test_phase])
# cassie_env.sim.set_qvel(qvel_phase[:, test_phase])
# cassie_env.phase = test_phase
# render_state = cassie_env.render()
# force_x = perturb_size * np.cos(0)
# force_y = perturb_size * np.sin(0)
# print("Perturb angle: {}\t Perturb size: {} N".format(np.degrees(-perturb_dir[dir_idx]), perturb_size))
# # Apply perturb (if time)
# while render_state:
#     if (not cassie_env.vis.ispaused()):
#         curr_time = cassie_env.sim.time()
#         if curr_time < perturb_duration:
#             cassie_env.vis.apply_force([force_x, force_y, 0, 0, 0, 0], perturb_body)
#         # Done perturbing, reset perturb_time and xfrc_applied
#         elif perturb_duration < curr_time < perturb_duration + wait_time:
#             cassie_env.vis.apply_force([0, 0, 0, 0, 0, 0], perturb_body)
#         else:
#             print("passed")
#             break           

#         # Get action
#         _, action = policy.act(state, True)
#         action = action.data.numpy()
#         state, reward, done, _ = cassie_env.step(action)
#         if cassie_env.sim.qpos()[2] < 0.4:
#             print("failed")
#             break
#         else:
#             state = torch.Tensor(state)
#     render_state = cassie_env.render()
#     time.sleep(dt / speedup)
# exit()

###### Vis all perturbations ######
render_state = cassie_env.render()
force_x = perturb_size * np.cos(0)
force_y = perturb_size * np.sin(0)
print("Perturb angle: {}\t Perturb size: {} N".format(np.degrees(-perturb_dir[dir_idx]), perturb_size))
while render_state:
    if (not cassie_env.vis.ispaused()):
        curr_time = cassie_env.sim.time()
        # Apply perturb (if time)
        if curr_time > perturb_time + wait_time:
            # Haven't perturbed for full time yet
            if curr_time < perturb_time + wait_time + perturb_duration:
                cassie_env.vis.apply_force([force_x, force_y, 0, 0, 0, 0], perturb_body)
            # Done perturbing, reset perturb_time and xfrc_applied
            else:
                cassie_env.vis.apply_force([0, 0, 0, 0, 0, 0], perturb_body)
                dir_idx += 1
                # Skip last direction, 0 is the same as 2*pi
                if dir_idx >= len(perturb_dir) - 1:
                    dir_idx = 0
                    perturb_size += 50
                force_x = perturb_size * np.cos(perturb_dir[dir_idx])
                force_y = perturb_size * np.sin(perturb_dir[dir_idx])
                print("Perturb angle: {}\t Perturb size: {} N".format(np.degrees(-perturb_dir[dir_idx]), perturb_size))
                perturb_time = curr_time

        # Get action
        _, action = policy.act(state, True)
        action = action.data.numpy()
        state, reward, done, _ = cassie_env.step(action)
        if cassie_env.sim.qpos()[2] < 0.4:
            state = torch.Tensor(cassie_env.reset_for_test())
            cassie_env.speed = 0.5
            cassie_env.side_speed = 0
            cassie_env.phase_add = 1
            perturb_time = 0
        else:
            state = torch.Tensor(state)
    render_state = cassie_env.render()
    time.sleep(dt / speedup)