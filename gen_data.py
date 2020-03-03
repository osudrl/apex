import torch
import sys, argparse, time
from cassie.cassie import CassieEnv_v2
import numpy as np
import ray
from functools import partial
import sys
import matplotlib.pyplot as plt
import pickle

@ray.remote
class gen_worker(object):
    def __init__(self, env_fn, policy):
        self.cassie_env = env_fn()
        self.policy = policy
        self.obs_dim = len(self.cassie_env.observation_space)
        self.phaselen = self.cassie_env.phaselen + 1

    # Simulates a single rollout of "num_steps" long at "speed" speed, will simulate "wait_cycles" number of cycles to stabilize
    # before apply perturbation of size "force" in "perturb_dir" direction on "perturb_body" for "perturb_len" time when the
    # robot is at phase "test_phase". 
    # Returns all the states observed in the rollout
    def sim_rollout(self, num_steps, wait_cycles, test_phase, speed, force, perturb_dir, perturb_body, perturb_duration, worker_id):

        # Check to make sure "wait_cycles" isn't too long. Need time to apply force after stabilize
        if self.phaselen*wait_cycles >= num_steps:
            print("Error: Too many wait_cycles. phaselen*wait_cycles is larger than num_steps")
            exit()

        start_t = time.time()
        data = np.zeros((num_steps, self.obs_dim))
        rewards = np.zeros(num_steps)
        force_x = force * np.cos(perturb_dir)
        force_y = force * np.sin(perturb_dir)
        state = self.cassie_env.reset_for_test()
        self.cassie_env.speed = speed
        # Simulate "wait_cycles" number of cycles to stablize before apply force
        # Should these be saved? Theoretically these will be the same everytime, so might bias the dataset
        for timestep in range(self.phaselen*wait_cycles):
            action = self.policy.forward(torch.Tensor(state)).detach().numpy()
            state, reward, done, _ = self.cassie_env.step(action)
            rewards[timestep] = reward
            data[timestep, :] = state
        # Simulate until at current testing phase
        for timestep in range(self.phaselen*wait_cycles, self.phaselen*wait_cycles+test_phase):
            action = self.policy.forward(torch.Tensor(state)).detach().numpy()
            state, reward, done, _ = self.cassie_env.step(action)
            rewards[timestep] = reward
            data[timestep, :] = state
        # Apply force
        force_start_t = self.cassie_env.sim.time()
        for timestep in range(self.phaselen*wait_cycles+test_phase, num_steps):
            if self.cassie_env.sim.time() < force_start_t + perturb_duration:
                self.cassie_env.sim.apply_force([force_x, force_y, 0, 0, 0, 0], perturb_body)
            else:
                self.cassie_env.sim.apply_force([0, 0, 0, 0, 0, 0], perturb_body)
            action = self.policy.forward(torch.Tensor(state)).detach().numpy()
            state, reward, done, _ = self.cassie_env.step(action)
            rewards[timestep] = reward
            data[timestep, :] = state
        
        return data, rewards, worker_id, time.time() - start_t

def gen_data_multi(env_fn, policy, num_steps, wait_cycles, speeds, forces, num_angles, perturb_body, perturb_duration, num_procs):

    total_start_t = time.time()
    temp_env = env_fn()
    obs_dim = len(temp_env.observation_space)
    if num_angles > 0:
        perturb_dir = -2*np.pi*(1/num_angles)*np.arange(num_angles)
    else:
        perturb_dir = []
    phaselen = temp_env.phaselen + 1
    num_speeds = len(speeds)
    num_forces = len(forces)

    # Form list of arguments
    args = []
    count = 0
    for i in range(num_speeds):
        # Initial rollout with no perturbs
        args.append((num_steps, wait_cycles, 0, speeds[i], 0, 0, perturb_body, perturb_duration))
        count += 1
        # Loop through forces
        for j in range(num_forces):
            # Loop through angles
            for k in range(num_angles):
                # Loop through phases
                for l in range(phaselen):
                    args.append((num_steps, wait_cycles, l, speeds[i], forces[j], perturb_dir[k], perturb_body, perturb_duration))
                    count += 1

    print("num args should be: ", num_speeds*(1+num_forces*num_angles*phaselen))
    print("Number of args: ", len(args))
    num_args = len(args)
    total_data = np.zeros((num_args*num_steps, obs_dim))
    total_rewards = np.zeros(num_args*num_steps)
    # total_data = None
    avg_time = None
    avg_iter_time = None
    done_count = 0

    # Make ray workers
    ray.shutdown()
    ray.init(include_webui=False, num_cpus=num_procs)#, temp_dir="~/ray_temp")
    workers = [gen_worker.remote(env_fn, policy) for i in range(num_procs)]
    print("made workers")

    # Launch initial jobs
    job_ids = [workers[i].sim_rollout.remote(*(args[i] + (i,))) for i in range(num_procs)]
    print("Started initial jobs")
    arg_ind = num_procs
    while done_count < num_args:
        iter_start_t = time.time()
        done_id = ray.wait(job_ids, num_returns=1, timeout=None)[0][0]
        done_data = ray.get(done_id)
        total_data[done_count*num_steps:(done_count+1)*num_steps, :] = done_data[0]
        total_rewards[done_count*num_steps:(done_count+1)*num_steps] = done_data[1]
        done_count += 1
        curr_id = done_data[2]
        if arg_ind < num_args: # Still jobs to submit
            job_ids.append(workers[curr_id].sim_rollout.remote(*(args[arg_ind]) + (curr_id,)))
            arg_ind += 1
        if avg_time is None:
            avg_time = done_data[3]
            avg_iter_time = time.time() - iter_start_t
        else:
            avg_time += (done_data[3] - avg_time) / done_count
            avg_iter_time += (time.time() - iter_start_t - avg_iter_time) / done_count
        print("\rFinished arg {} out of {}\tAvg sim time: {:.4f}\tTime left: {:.4f}".format(done_count, 
                    num_args, avg_time, (num_args - done_count)*(avg_iter_time)), end='')
        # exit()
    print("")
    np.savez("./5b75b3-seed0_gen_data.npz", total_data=total_data, total_rewards=total_rewards)
    print("Total time: ", time.time() - total_start_t)


def gen_data(cassie_env, policy, num_steps, wait_cycles, speeds, forces, num_angles, perturb_body, perturb_duration):

    obs_dim = len(cassie_env.observation_space)
    perturb_dir = -2*np.pi*(1/num_angles)*np.arange(num_angles)
    phaselen = cassie_env.phaselen + 1
    num_speeds = len(speeds)
    num_forces = len(forces)

    if phaselen*wait_cycles >= num_steps:
        print("Error: Too many wait_cycles. phaselen*wait_cycles is larger than num_steps")
        exit()

    data = np.zeros(((num_speeds*num_forces*num_angles*phaselen + 1)*num_steps, obs_dim))

    for i in range(num_speeds):
        start_t = time.time()
        # Initial rollout with no perturbs
        state = cassie_env.reset_for_test()
        cassie_env.speed = speeds[i]
        for j in range(num_steps):
            action = policy.forward(torch.Tensor(state)).detach().numpy()
            state, reward, done, _ = cassie_env.step(action)
            data[j+i*(num_forces*num_angles*phaselen+1), :] = state
        print("num_steps sim time: ", time.time() - start_t)
        # Loop through forces
        for j in range(num_forces):
            # Loop through angles
            for k in range(num_angles):
                force_x = forces[j] * np.cos(perturb_dir[k])
                force_y = forces[j] * np.sin(perturb_dir[k])
                # Loop through phases
                for l in range(phaselen):
                    state = cassie_env.reset_for_test()
                    cassie_env.speed = speeds[i]
                    ind_offset = num_steps+l*num_steps+k*phaselen*num_steps+j*num_angles*phaselen*num_steps+i*(num_forces*num_angles*phaselen+1)*num_steps
                    # Simulate "wait_cycles" number of cycles to stablize before apply force
                    # Should these be saved? Theoretically these will be the same everytime, so might bias the dataset
                    for timestep in range(phaselen*wait_cycles):
                        action = policy.forward(torch.Tensor(state)).detach().numpy()
                        state, reward, done, _ = cassie_env.step(action)
                        data[timestep+ind_offset, :] = state
                    # Simulate until at current testing phase
                    for timestep in range(phaselen*wait_cycles, phaselen*wait_cycles+l):
                        action = policy.forward(torch.Tensor(state)).detach().numpy()
                        state, reward, done, _ = cassie_env.step(action)
                        data[timestep+ind_offset, :] = state
                    # Apply force
                    start_time = cassie_env.sim.time()
                    for timestep in range(phaselen*wait_cycles+l, num_steps):
                        if cassie_env.sim.time() < start_time + perturb_duration:
                            cassie_env.sim.apply_force([force_x, force_y, 0, 0, 0, 0], perturb_body)
                        else:
                            cassie_env.sim.apply_force([0, 0, 0, 0, 0, 0], perturb_body)
                        action = policy.forward(torch.Tensor(state)).detach().numpy()
                        state, reward, done, _ = cassie_env.step(action)
                        data[timestep+ind_offset, :] = state
        print("sim time for single speed: ", time.time() - start_t)

    np.save("./test_obs_data.npy", data)

def vis_rollout(cassie_env, policy, num_steps, wait_cycles, test_phase, speed, force, perturb_dir, perturb_body, perturb_duration):

    phaselen = cassie_env.phaselen + 1

    # Check to make sure "wait_cycles" isn't too long. Need time to apply force after stabilize
    if phaselen*wait_cycles >= num_steps:
        print("Error: Too many wait_cycles. phaselen*wait_cycles is larger than num_steps")
        exit()

    force_x = force * np.cos(perturb_dir)
    force_y = force * np.sin(perturb_dir)
    state = cassie_env.reset_for_test()
    cassie_env.speed = speed
    render_state = cassie_env.render()
    timestep = 0
    start_time = None
    render_done = False
    while render_state and not render_done:
        if (not cassie_env.vis.ispaused()):
            # Simulate "wait_cycles" number of cycles to stablize before apply force
            if timestep < phaselen*wait_cycles + test_phase:
                cassie_env.sim.apply_force([0, 0, 0, 0, 0, 0], perturb_body)
            elif timestep == phaselen*wait_cycles + test_phase:
                start_time = cassie_env.sim.time()
                cassie_env.sim.apply_force([force_x, force_y, 0, 0, 0, 0], perturb_body)
                print("start force. phase is ", cassie_env.phase)
            elif (start_time is not None and cassie_env.sim.time() < start_time + perturb_duration):
                cassie_env.sim.apply_force([force_x, force_y, 0, 0, 0, 0], perturb_body)
                print("applying force")
            elif timestep < num_steps and (start_time is not None and cassie_env.sim.time() > start_time + perturb_duration):
                cassie_env.sim.apply_force([0, 0, 0, 0, 0, 0], perturb_body)
            else:
                print("done")
                render_done = True

            action = policy.forward(torch.Tensor(state)).detach().numpy()
            state, reward, done, _ = cassie_env.step(action)
            timestep += 1
        render_state = cassie_env.render()
        time.sleep(0.05/3)
    

# Check data
# data_dict = np.load("./5b75b3-seed0_gen_data.npz")
# obs_data = data_dict["total_data"]
# reward_data = data_dict["total_rewards"]
# print("reward shape: ", reward_data.shape)
# print("max reward: ", np.max(reward_data))
# print("min reward: ", np.min(reward_data))
# plt.hist(reward_data)
# plt.savefig("./5b75b3-seed0_gen_data_rewards.png")
# exit()

# Load policy and env
# policy = torch.load("./trained_models/new_policies/nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2_actor.pt")
exp_dir = "./trained_models/ppo/Cassie-v0/5b75b3-seed0/"
policy = torch.load(exp_dir + "actor.pt")

# policy.eval()
run_args = pickle.load(open(exp_dir + "experiment.pkl", "rb"))
cassie_env = CassieEnv_v2(traj=run_args.traj, state_est=run_args.state_est, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, history=run_args.history)
env_fn = partial(CassieEnv_v2, traj=run_args.traj, state_est=run_args.state_est, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, history=run_args.history)
# cassie_env = CassieEnv_v2(clock_based=True, state_est=False, no_delta=True)
# env_fn = partial(CassieEnv_v2, clock_based=True, state_est=False, no_delta=True)

# Set data generation parameters
num_steps = 300
wait_cycles = 2
num_angles = 0
speeds = np.linspace(0, 1, 11)
forces = []
perturb_body = "cassie-pelvis"
perturb_duration = 0.0

# NOTE: There are no checks made for whether Cassie fell down or not. If the policy failed for some reason, data will
# still be saved. For this reason, make sure that the policy can resist the specified forces
# gen_data(cassie_env, policy, num_steps, wait_cycles, speeds, forces, num_angles, perturb_body, perturb_duration)
vis_rollout(cassie_env, policy, 300, 2, 0, .5, 0, 0, perturb_body, perturb_duration)
# gen_data_multi(env_fn, policy, num_steps, wait_cycles, speeds, forces, num_angles, perturb_body, perturb_duration, 11)
