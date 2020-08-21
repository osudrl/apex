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
import copy
from functools import partial

@ray.remote
class perturb_worker(object):
    def __init__(self, id_num, env_fn, policy, num_angles, wait_time, perturb_duration, start_size, perturb_incr, perturb_body):
        self.id_num = id_num
        self.cassie_env = env_fn()
        self.policy = copy.deepcopy(policy)
        self.perturb_dirs = -2*np.pi*np.linspace(0, 1, num_angles+1)
        self.wait_time = wait_time
        self.perturb_duration = perturb_duration
        self.start_size = start_size
        self.perturb_incr = perturb_incr
        self.perturb_body = perturb_body
        self.num_phases = self.cassie_env.phaselen + 1

    @torch.no_grad()
    def reset_to_phase(self, phase):
        state = torch.Tensor(self.cassie_env.reset_for_test(full_reset=True))
        self.cassie_env.speed = 0.5
        for i in range(2*self.num_phases):
            action = self.policy(state, True)
            action = action.data.numpy()
            state, reward, done, _ = self.cassie_env.step(action)
            state = torch.Tensor(state)
        for i in range(phase):
            action = self.policy(state, True)
            action = action.data.numpy()
            state, reward, done, _ = self.cassie_env.step(action)
            state = torch.Tensor(state)
        return state

    @torch.no_grad()
    # Runs a perturbation for a single inputted angle and phase
    def perturb_test_angle(self, dir_ind, phase):
        eval_start = time.time()
        angle = self.perturb_dirs[dir_ind]
        # print("Testing angle {} ({} out of {}) for phase {}".format(-angles[i], i+1, num_angles, j))
        curr_size = self.start_size - self.perturb_incr
        done = False
        while not done:
            curr_size += self.perturb_incr
            state = self.reset_to_phase(phase)
            # Apply perturb
            force_x = curr_size * np.cos(angle)
            force_y = curr_size * np.sin(angle)
            curr_time = self.cassie_env.sim.time()
            start_t = curr_time
            while curr_time < start_t + self.perturb_duration:
                self.cassie_env.sim.apply_force([force_x, force_y, 0, 0, 0, 0], self.perturb_body)
                action = self.policy(state, True)
                action = action.data.numpy()
                state, reward, done, _ = self.cassie_env.step(action)
                state = torch.Tensor(state)
                curr_time = self.cassie_env.sim.time()
            # Done perturbing, reset xfrc_applied
            self.cassie_env.sim.apply_force([0, 0, 0, 0, 0, 0], self.perturb_body)
            start_t = curr_time
            while curr_time < start_t + self.wait_time:
                action = self.policy(state, True)
                action = action.data.numpy()
                state, reward, done, _ = self.cassie_env.step(action)
                state = torch.Tensor(state)
                curr_time = self.cassie_env.sim.time()
                if self.cassie_env.sim.qpos()[2] < 0.4:  # Failed, reset and record force
                    done = True
                    break
        max_force = curr_size - self.perturb_incr
        # print("max force: ", curr_size - perturb_incr)
        return self.id_num, dir_ind, phase, max_force, time.time() - eval_start


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

def compute_perturbs_multi(env_fn, policy, wait_time=4, perturb_duration=0.2, perturb_size=100, perturb_incr=10, perturb_body="cassie-pelvis", num_angles=4, num_procs=4):
    start_t = time.time()
    bar_width = 30
    # Check to make sure don't make too many workers for the given
    temp_env = env_fn()
    num_phases = temp_env.phaselen + 1 
    if num_procs > num_angles * num_phases:
        num_procs = num_angles * num_phases
    # Make all args
    args = [(i, j) for i in range(num_angles) for j in range(num_phases)]
    total_data = np.zeros((num_angles, num_phases))
    num_args = num_angles * num_phases
    
    # Make and start all workers
    print("Using {} processes".format(num_procs))
    ray.init(num_cpus=num_procs)
    workers = [perturb_worker.remote(i, env_fn, policy, num_angles, wait_time, perturb_duration, perturb_size, perturb_incr, perturb_body) for i in range(num_procs)]
    print("made workers")
    eval_start = time.time()
    result_ids = [workers[i].perturb_test_angle.remote(*args[i]) for i in range(num_procs)]
    print("started workers")
    curr_arg_ind = num_procs
    sys.stdout.write(progress_bar(0, num_args, bar_width, 0, 0))
    sys.stdout.flush()
    avg_eval_time = 0
    count = 0
    while result_ids:
        done_id = ray.wait(result_ids, num_returns=1, timeout=None)[0][0]
        worker_id, dir_ind, phase, max_force, eval_time = ray.get(done_id)
        total_data[dir_ind, phase] = max_force
        result_ids.remove(done_id)
        if curr_arg_ind < num_args:
            result_ids.append(workers[worker_id].perturb_test_angle.remote(*args[curr_arg_ind]))
        curr_arg_ind += 1
        count += 1
        avg_eval_time += (eval_time - avg_eval_time) / count

        sys.stdout.write("\r{}".format(progress_bar(count, num_args, bar_width, (time.time()-eval_start), avg_eval_time*num_args/num_procs)))
        sys.stdout.flush()

    print("")
    print("total time: ", time.time() - start_t)
    ray.shutdown()
    return total_data

def progress_bar(curr_ind, total_ind, bar_width, elapsed_time, est_total_time):
    num_bar = int((curr_ind / total_ind) // (1/bar_width))
    num_space = int(bar_width - num_bar)
    outstring = "[{}]".format("-"*num_bar + " "*num_space)
    outstring += " {:.2f}%".format(curr_ind / total_ind * 100)
    if elapsed_time == 0:
        time_left = "N/A"
        outstring += ", {:.1f}s elapsed, {}s ({}) left".format(elapsed_time, time_left, time_left)
    else:
        time_left = elapsed_time/curr_ind*(total_ind-curr_ind)
        outstring += ", {:.1f}s elapsed, {:.1f}s ({:.1f}) left".format(elapsed_time, time_left, est_total_time-elapsed_time)
    return outstring

def plot_perturb(filename, plotname, max_force):
    data = np.load(filename)
    data = np.mean(data, axis=1)
    num_angles = len(data)
    
    num_cells = 100
    fig, ax1 = plt.subplots(subplot_kw=dict(projection='polar'))
    ax1.patch.set_alpha(0)
    offset = 2*np.pi/100/2
    theta, r = np.mgrid[0-offset:2*np.pi-offset:complex(0,num_angles+1), 0:max_force:complex(0, num_cells)]
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
    # plt.show()
    plt.savefig(plotname)

################################
##### DEPRACATED FUNCTIONS #####
################################

@ray.remote
@torch.no_grad()
def perturb_worker_old(env_fn, qpos_phase, qvel_phase, policy, angles, wait_time, perturb_duration, perturb_size, perturb_incr, perturb_body, worker_id):
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

def compute_perturbs_multi_old(env_fn, policy, wait_time=4, perturb_duration=0.2, perturb_size=100, perturb_incr=10, perturb_body="cassie-pelvis", num_angles=4, num_procs=4):
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
