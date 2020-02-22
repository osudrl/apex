import sys
sys.path.append("..") # Adds higher directory to python modules path.

import argparse
import pickle

import numpy as np
import torch
import time
import copy

from cassie import CassieEnv

# Will reset the env to the given phase by reset_for_test, and then
# simulating 2 cycle then to the given phase
@torch.no_grad()
def reset_to_phase(env, policy, phase):
    state = torch.Tensor(cassie_env.reset_for_test())
    for i in range(2*(env.phaselen + 1)):
        action = policy.act(state, True)
        action = action.data.numpy()
        state, reward, done, _ = cassie_env.step(action)
        state = torch.Tensor(state)
    for i in range(phase):
        action = policy.act(state, True)
        action = action.data.numpy()
        state, reward, done, _ = cassie_env.step(action)
        state = torch.Tensor(state)

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=None, help="path to folder containing policy and run details")
args = parser.parse_args()
run_args = pickle.load(open(args.path + "experiment.pkl", "rb"))

# RUN_NAME = "7b7e24-seed0"
# POLICY_PATH = "../trained_models/ppo/Cassie-v0/" + RUN_NAME + "/actor.pt"

# Load environment and policy
# env_fn = partial(CassieEnv_speed_no_delta_neutral_foot, "walking", clock_based=True, state_est=True)
cassie_env = CassieEnv(traj=run_args.traj, clock_based=run_args.clock_based, state_est=run_args.state_est, dynamics_randomization=run_args.dyn_random)
policy = torch.load(args.path + "actor.pt")

state = torch.Tensor(cassie_env.reset_for_test())
# cassie_env.sim.step_pd(self.u)
cassie_env.speed = 0.5
cassie_env.phase_add = 1
num_steps = cassie_env.phaselen + 1
# Simulate for "wait_time" first to stabilize
for i in range(num_steps*2):
    action = policy(state, True)
    action = action.data.numpy()
    state, reward, done, _ = cassie_env.step(action)
    state = torch.Tensor(state)
curr_time = cassie_env.sim.time()
start_t = curr_time
sim_t = time.time()
while curr_time < start_t + 4:
    action = policy(state, True)
    action = action.data.numpy()
    state, reward, done, _ = cassie_env.step(action)
    state = torch.Tensor(state)
    curr_time = cassie_env.sim.time()
print("sim time: ", time.time() - sim_t)
exit()
qpos_phase = np.zeros((35, num_steps))
qvel_phase = np.zeros((32, num_steps))
action_phase = np.zeros((10, num_steps))
cassie_state_phase = [copy.deepcopy(cassie_env.cassie_state)]
# print("phase: ", cassie_env.phase)
qpos_phase[:, 0] = cassie_env.sim.qpos()
qvel_phase[:, 0] = cassie_env.sim.qvel()
for i in range(num_steps-1):
    action = policy.act(state, True)
    action = action.data.numpy()
    action_phase[:, i] = action
    state, reward, done, _ = cassie_env.step(action)
    state = torch.Tensor(state)
    # print("phase: ", cassie_env.phase)
    qpos_phase[:, i+1] = cassie_env.sim.qpos()
    qvel_phase[:, i+1] = cassie_env.sim.qvel()
    cassie_state_phase.append(copy.deepcopy(cassie_env.cassie_state))

action = policy.act(state, True)
action = action.data.numpy()
action_phase[:, -1] = action
state = torch.Tensor(cassie_env.reset_for_test())

cassie_env.speed = 0.5
cassie_env.phase_add = 1
wait_time = 4
dt = 0.05
speedup = 3
perturb_time = 2
perturb_duration = 0.2
perturb_size = 170
perturb_dir = -2*np.pi*np.linspace(0, 1, 5)  # Angles from straight forward to apply force
perturb_body = "cassie-pelvis"
dir_idx = 0

###### Vis a single Perturbation for a given phase ######
test_phase = 0
reset_to_phase(cassie_env, policy, test_phase)
# cassie_env.sim.set_qpos(qpos_phase[:, test_phase])
# cassie_env.sim.set_qvel(qvel_phase[:, test_phase])
# cassie_env.cassie_state = cassie_state_phase[test_phase]
# cassie_env.sim.set_cassie_state(cassie_state_phase[test_phase])
# cassie_env.phase = test_phase
# state, reward, done, _ = cassie_env.step(action_phase[:, test_phase-1])
# state = torch.Tensor(state)
render_state = cassie_env.render()
force_x = perturb_size * np.cos(0)
force_y = perturb_size * np.sin(0)
print("Perturb angle: {}\t Perturb size: {} N".format(np.degrees(-perturb_dir[dir_idx]), perturb_size))
# Apply perturb (if time)
start_t = cassie_env.sim.time()
while render_state:
    if (not cassie_env.vis.ispaused()):
        curr_time = cassie_env.sim.time()
        if curr_time < start_t+perturb_duration:
            cassie_env.vis.apply_force([force_x, force_y, 0, 0, 0, 0], perturb_body)
        # Done perturbing, reset perturb_time and xfrc_applied
        elif start_t+perturb_duration < curr_time < start_t+perturb_duration + wait_time:
            # print("curr time: ", curr_time)
            cassie_env.vis.apply_force([0, 0, 0, 0, 0, 0], perturb_body)
        else:
            # pass
            print("passed")
            break           

        # Get action
        action = policy.act(state, True)
        action = action.data.numpy()
        state, reward, done, _ = cassie_env.step(action)
        if cassie_env.sim.qpos()[2] < 0.4:
            print("failed")
            break
        else:
            state = torch.Tensor(state)
    render_state = cassie_env.render()
    time.sleep(dt / speedup)
exit()

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
                print("phase: ", cassie_env.phase)
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
        action = policy.act(state, True)
        action = action.data.numpy()
        state, reward, done, _ = cassie_env.step(action)
        if cassie_env.sim.qpos()[2] < 0.4:
            state = torch.Tensor(cassie_env.reset_for_test())
            cassie_env.speed = 0.5
            cassie_env.phase_add = 1
            perturb_time = 0
        else:
            state = torch.Tensor(state)
    render_state = cassie_env.render()
    time.sleep(dt / speedup)