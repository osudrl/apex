import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import time
import math
import random
import ray
import copy, sys
from functools import partial

# Quaternion utility functions. Due to python relative imports and directory structure can't cleanly use cassie.quaternion_function
def inverse_quaternion(quaternion):
	result = np.copy(quaternion)
	result[1:4] = -result[1:4]
	return result

def quaternion_product(q1, q2):
	result = np.zeros(4)
	result[0] = q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]-q1[3]*q2[3]
	result[1] = q1[0]*q2[1]+q2[0]*q1[1]+q1[2]*q2[3]-q1[3]*q2[2]
	result[2] = q1[0]*q2[2]-q1[1]*q2[3]+q1[2]*q2[0]+q1[3]*q2[1]
	result[3] = q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1]+q1[3]*q2[0]
	return result

def rotate_by_quaternion(vector, quaternion):
	q1 = np.copy(quaternion)
	q2 = np.zeros(4)
	q2[1:4] = np.copy(vector)
	q3 = inverse_quaternion(quaternion)
	q = quaternion_product(q2, q3)
	q = quaternion_product(q1, q)
	result = q[1:4]
	return result

def euler2quat(z=0, y=0, x=0):
    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    result =  np.array([
             cx*cy*cz - sx*sy*sz,
             cx*sy*sz + cy*cz*sx,
             cx*cz*sy - sx*cy*sz,
             cx*cy*sz + sx*cz*sy])
    if result[0] < 0:
    	result = -result
    return result

@ray.remote
class eval_worker(object):
    def __init__(self, id_num, env_fn, policy, num_steps, max_speed, min_speed):
        self.id_num = id_num
        self.cassie_env = env_fn()
        self.policy = copy.deepcopy(policy)
        self.num_steps = num_steps
        self.max_speed = max_speed
        self.min_speed = min_speed

    @torch.no_grad()
    def run_test(self, speed_schedule, orient_schedule):
        start_t = time.time()
        save_data = np.zeros(6)
        state = torch.Tensor(self.cassie_env.reset_for_test(full_reset=True))
        self.cassie_env.speed = 0.5
        self.cassie_env.side_speed = 0
        self.cassie_env.phase_add = 1
        num_commands = len(orient_schedule)
        count = 0
        orient_ind = 0
        speed_ind = 1 
        orient_add = 0
        passed = 1
        while not (speed_ind == num_commands and orient_ind == num_commands and count == self.num_steps) and passed:
            # Update speed command
            if count == self.num_steps:
                count = 0
                self.cassie_env.speed = speed_schedule[speed_ind]
                self.cassie_env.speed = np.clip(self.cassie_env.speed, self.min_speed, self.max_speed)
                if self.cassie_env.speed > 1.4:
                    self.cassie_env.phase_add = 1.5
                else:
                    self.cassie_env.phase_add = 1
                speed_ind += 1
            # Update orientation command
            elif count == self.num_steps // 2:
                orient_add += orient_schedule[orient_ind]
                orient_ind += 1
            # Update orientation
            # TODO: Make update orientation function in each env to this will work with an abitrary environment
            quaternion = euler2quat(z=orient_add, y=0, x=0)
            iquaternion = inverse_quaternion(quaternion)
            curr_orient = state[1:5]
            curr_transvel = state[15:18]

            new_orient = quaternion_product(iquaternion, curr_orient)
            if new_orient[0] < 0:
                new_orient = -new_orient
            new_translationalVelocity = rotate_by_quaternion(curr_transvel, iquaternion)
            state[1:5] = torch.FloatTensor(new_orient)
            state[15:18] = torch.FloatTensor(new_translationalVelocity)

            # Get action
            action = self.policy(state, True)
            action = action.data.numpy()
            state, reward, done, _ = self.cassie_env.step(action)
            state = torch.Tensor(state)
            if self.cassie_env.sim.qpos()[2] < 0.4:
                passed = 0
            count += 1
        if passed:
            save_data[0] = passed
            save_data[1] = -1
        else:
            save_data[:] = np.array([passed, count//(self.num_steps//2), self.cassie_env.speed, orient_add,\
                        self.cassie_env.speed-speed_schedule[max(0, speed_ind-2)], orient_schedule[orient_ind-1]])
        return self.id_num, save_data, time.time() - start_t

def eval_commands_multi(env_fn, policy, num_steps=200, num_commands=4, max_speed=3, min_speed=0, num_iters=4, num_procs=4, filename="test_eval_command.npy"):
    start_t1 = time.time()
    ray.init(num_cpus=num_procs)
    total_data = np.zeros((num_iters, 6))
    # Make all args
    all_speed_schedule = np.zeros((num_iters, num_commands))
    all_orient_schedule = np.zeros((num_iters, num_commands))
    for i in range(num_iters):
        all_speed_schedule[i, 0] = 0.5
        for j in range(num_commands-1):
            speed_add = random.choice([-1, 1])*random.uniform(0.4, 1.3)
            if all_speed_schedule[i, j] + speed_add < min_speed or all_speed_schedule[i, j] + speed_add > max_speed:
                speed_add *= -1
            all_speed_schedule[i, j+1] = all_speed_schedule[i, j] + speed_add
        orient_schedule = np.random.uniform(np.pi/6, np.pi/3, num_commands)
        orient_sign = np.random.choice((-1, 1), num_commands)
        all_orient_schedule[i, :] = orient_schedule * orient_sign
    # Make and start eval workers
    workers = [eval_worker.remote(i, env_fn, policy, num_steps, max_speed, min_speed) for i in range(num_procs)]
    eval_ids = [workers[i].run_test.remote(all_speed_schedule[i, :], all_orient_schedule[i, :]) for i in range(num_procs)]
    print("started workers")
    curr_arg_ind = num_procs
    curr_data_ind = 0
    bar_width = 30
    sys.stdout.write(progress_bar(0, num_iters, bar_width, 0))
    sys.stdout.flush()
    eval_start = time.time()
    while curr_arg_ind < num_iters:
        done_id = ray.wait(eval_ids, num_returns=1, timeout=None)[0][0]
        worker_id, data, eval_time = ray.get(done_id)
        total_data[curr_data_ind, :] = data
        eval_ids.remove(done_id)
        eval_ids.append(workers[worker_id].run_test.remote(all_speed_schedule[curr_arg_ind, :], all_orient_schedule[curr_arg_ind, :]))
        curr_arg_ind += 1
        curr_data_ind += 1

        sys.stdout.write("\r{}".format(progress_bar(curr_data_ind, num_iters, bar_width, (time.time()-eval_start))))
        sys.stdout.flush()
    
    result = ray.get(eval_ids)
    for ret_tuple in result:
        total_data[curr_data_ind, :] = ret_tuple[1]
        curr_data_ind += 1
    sys.stdout.write("\r{}".format(progress_bar(num_iters, num_iters, bar_width, time.time()-eval_start)))
    print("")
    print("Got all results")
    np.save(filename, total_data)
    print("total time: ", time.time() - start_t1)
    ray.shutdown()

def progress_bar(curr_ind, total_ind, bar_width, elapsed_time):
    num_bar = int((curr_ind / total_ind) // (1/bar_width))
    num_space = int(bar_width - num_bar)
    outstring = "[{}]".format("-"*num_bar + " "*num_space)
    outstring += " {:.2f}% complete".format(curr_ind / total_ind * 100)
    if elapsed_time == 0:
        time_left = "N/A"
        outstring += " {:.1f} elapsed, {} left".format(elapsed_time, time_left)
    else:
        time_left = elapsed_time/curr_ind*(total_ind-curr_ind)
        outstring += " {:.1f} elapsed, {:.1f} left".format(elapsed_time, time_left)
    return outstring

def report_stats(filename):
    data = np.load(filename)
    num_iters = data.shape[0]
    pass_rate = np.sum(data[:, 0]) / num_iters
    success_inds = np.where(data[:, 0] == 1)[0]
    # data[success_inds, 1] = -1
    speed_fail_inds = np.where(data[:, 1] == 0)[0]
    orient_fail_inds = np.where(data[:, 1] == 1)[0]
    print("pass rate: ", pass_rate)
    # print("speed failure: ", data[speed_fail_inds, 4])
    # print("orient failure: ", data[orient_fail_inds, 5])
    speed_change = data[speed_fail_inds, 4]
    orient_change = data[orient_fail_inds, 5]
    speed_neg_inds = np.where(speed_change < 0)
    speed_pos_inds = np.where(speed_change > 0)
    orient_neg_inds = np.where(orient_change < 0)
    orient_pos_inds = np.where(orient_change > 0)
    print("Number of speed failures: ", len(speed_fail_inds))
    print("Number of orient failures: ", len(orient_fail_inds))
    if len(speed_fail_inds) == 0:
        avg_pos_speed = "N/A"
        avg_neg_speed = "N/A"
    else:
        avg_pos_speed = np.mean(speed_change[speed_pos_inds])
        avg_neg_speed = np.mean(speed_change[speed_neg_inds])
    if len(orient_fail_inds) == 0:
        avg_pos_orient = "N/A"
        avg_neg_orient = "N/A"
    else:
        avg_pos_orient = np.mean(orient_change[orient_pos_inds])
        avg_neg_orient = np.mean(orient_change[orient_neg_inds])
    print("avg pos speed failure: ", avg_pos_speed)
    print("avg neg speed failure: ", avg_neg_speed)
    print("avg pos orient failure: ", avg_pos_orient)
    print("avg neg orient failure: ", avg_neg_orient)


@torch.no_grad()
def eval_commands(cassie_env, policy, num_steps=200, num_commands=2, max_speed=3, min_speed=0, num_iters=1):
    # save_data will hold whether passed or not (1 or 0), whether orient command or speed command caused failure (1, 0),
    # speed and orient command at failure, and speed and orient change at failure
    save_data = np.zeros((num_iters, 6))
    start_t = time.time()
    for j in range(num_iters):
        state = torch.Tensor(cassie_env.reset_for_test())
        cassie_env.speed = 0.5
        cassie_env.side_speed = 0
        cassie_env.phase_add = 1
        speed_schedule = [0.5]
        for i in range(num_commands-1):
            speed_add = random.choice([-1, 1])*random.uniform(0.4, 1.3)
            if speed_schedule[i] + speed_add < min_speed or speed_schedule[i] + speed_add > max_speed:
                speed_add *= -1
            speed_schedule.append(speed_schedule[i] + speed_add)
        orient_schedule = np.random.uniform(np.pi/6, np.pi/3, num_commands)
        orient_sign = np.random.choice((-1, 1), num_commands)
        orient_schedule = orient_schedule * orient_sign
        # print("Speed schedule: ", speed_schedule)
        # print("Orient schedule: ", orient_schedule)
        count = 0
        orient_ind = 0
        speed_ind = 1 
        orient_add = 0
        passed = 1
        while not (speed_ind == num_commands and orient_ind == num_commands and count == num_steps) and passed:
            if count == num_steps:
                count = 0
                cassie_env.speed = speed_schedule[speed_ind]
                cassie_env.speed = np.clip(cassie_env.speed, min_speed, max_speed)
                if cassie_env.speed > 1.4:
                    cassie_env.phase_add = 1.5
                else:
                    cassie_env.phase_add = 1
                speed_ind += 1
                # print("Current speed: ", cassie_env.speed, speed_ind)
            elif count == num_steps // 2:
                orient_add += orient_schedule[orient_ind]
                orient_ind += 1
                # print("Current orient add: ", orient_add, orient_ind)
            # Update orientation
            quaternion = euler2quat(z=orient_add, y=0, x=0)
            iquaternion = inverse_quaternion(quaternion)
            curr_orient = state[1:5]
            curr_transvel = state[15:18]

            new_orient = quaternion_product(iquaternion, curr_orient)
            if new_orient[0] < 0:
                new_orient = -new_orient
            new_translationalVelocity = rotate_by_quaternion(curr_transvel, iquaternion)
            state[1:5] = torch.FloatTensor(new_orient)
            state[15:18] = torch.FloatTensor(new_translationalVelocity)

            # Get action
            action = policy(state, True)
            action = action.data.numpy()
            state, reward, done, _ = cassie_env.step(action)
            state = torch.Tensor(state)
            if cassie_env.sim.qpos()[2] < 0.4:
                # print("Failed")
                passed = 0
            count += 1
        if passed:
            # print("passed")
            save_data[j, 0] = passed
            save_data[j, 1] = -1
        else:
            # print("didnt pass")
            save_data[j, :] = np.array([passed, count//(num_steps//2), cassie_env.speed, orient_add,\
                        cassie_env.speed-speed_schedule[max(0, speed_ind-2)], orient_schedule[orient_ind-1]])
    print("time: ", time.time() - start_t)
    return save_data

def vis_commands(cassie_env, policy, num_steps=200, num_commands=4, max_speed=1, min_speed=0):
    state = torch.Tensor(cassie_env.reset_for_test())

    cassie_env.speed = 0.5
    cassie_env.side_speed = 0
    cassie_env.phase_add = 1
    # orient_schedule = np.pi/4*np.arange(8)
    # speed_schedule = np.random.uniform(-1.5, 1.5, 4)
    speed_schedule = [0.5]
    for i in range(num_commands-1):
        speed_add = random.choice([-1, 1])*random.uniform(0.4, 1.3)
        if speed_schedule[i] + speed_add < min_speed or speed_schedule[i] + speed_add > max_speed:
            speed_add *= -1
        speed_schedule.append(speed_schedule[i] + speed_add)
    orient_schedule = np.random.uniform(np.pi/6, np.pi/3, num_commands)
    orient_sign = np.random.choice((-1, 1), num_commands)
    orient_schedule = orient_schedule * orient_sign
    print("Speed schedule: ", speed_schedule)
    print("Orient schedule: ", orient_schedule)
    dt = 0.05
    speedup = 3
    count = 0
    orient_ind = 0
    speed_ind = 0 
    orient_add = 0
    # print("Current orient add: ", orient_add)

    render_state = cassie_env.render()
    with torch.no_grad():
        while render_state:
            if (not cassie_env.vis.ispaused()):
                # orient_add = orient_schedule[math.floor(count/num_steps)]
                if count == num_steps:
                    count = 0
                    speed_ind += 1
                    if speed_ind >= len(speed_schedule):
                        print("speed Done")
                        exit()
                    cassie_env.speed = speed_schedule[speed_ind]
                    cassie_env.speed = np.clip(cassie_env.speed, 0, 3)
                    if cassie_env.speed > 1.4:
                        cassie_env.phase_add = 1.5
                    print("Current speed: ", cassie_env.speed)
                elif count == num_steps // 2:
                    orient_ind += 1
                    if orient_ind >= len(orient_schedule):
                        print("orient Done")
                        exit()
                    orient_add += orient_schedule[orient_ind]
                    print("Current orient add: ", orient_add)
                # Update orientation
                quaternion = euler2quat(z=orient_add, y=0, x=0)
                iquaternion = inverse_quaternion(quaternion)
                curr_orient = state[1:5]
                curr_transvel = state[15:18]

                new_orient = quaternion_product(iquaternion, curr_orient)
                if new_orient[0] < 0:
                    new_orient = -new_orient
                new_translationalVelocity = rotate_by_quaternion(curr_transvel, iquaternion)
                state[1:5] = torch.FloatTensor(new_orient)
                state[15:18] = torch.FloatTensor(new_translationalVelocity)

                # Get action
                action = policy(state, True)
                action = action.data.numpy()
                state, reward, done, _ = cassie_env.step(action)
                if cassie_env.sim.qpos()[2] < 0.4:
                    print("Failed")
                    exit()
                else:
                    state = torch.Tensor(state)
                count += 1
                
            render_state = cassie_env.render()
            time.sleep(dt / speedup)

################################
##### DEPRACATED FUNCTIONS #####
################################

@ray.remote
@torch.no_grad()
def eval_commands_worker(env_fn, policy, num_steps, num_commands, max_speed, min_speed, num_iters):
    cassie_env = env_fn()
    # save_data will hold whether passed or not (1 or 0), whether orient command or speed command caused failure (1, 0),
    # speed and orient command at failure, and speed and orient change at failure
    save_data = np.zeros((num_iters, 6))
    start_t = time.time()
    for j in range(num_iters):
        state = torch.Tensor(cassie_env.reset_for_test())
        cassie_env.speed = 0.5
        cassie_env.side_speed = 0
        cassie_env.phase_add = 1
        speed_schedule = [0.5]
        for i in range(num_commands-1):
            speed_add = random.choice([-1, 1])*random.uniform(0.4, 1.3)
            if speed_schedule[i] + speed_add < min_speed or speed_schedule[i] + speed_add > max_speed:
                speed_add *= -1
            speed_schedule.append(speed_schedule[i] + speed_add)
        orient_schedule = np.random.uniform(np.pi/6, np.pi/3, num_commands)
        orient_sign = np.random.choice((-1, 1), num_commands)
        orient_schedule = orient_schedule * orient_sign
        count = 0
        orient_ind = 0
        speed_ind = 1 
        orient_add = 0
        passed = 1
        while not (speed_ind == num_commands and orient_ind == num_commands and count == num_steps) and passed:
            if count == num_steps:
                count = 0
                cassie_env.speed = speed_schedule[speed_ind]
                cassie_env.speed = np.clip(cassie_env.speed, min_speed, max_speed)
                if cassie_env.speed > 1.4:
                    cassie_env.phase_add = 1.5
                else:
                    cassie_env.phase_add = 1
                speed_ind += 1
            elif count == num_steps // 2:
                orient_add += orient_schedule[orient_ind]
                orient_ind += 1
            # Update orientation
            quaternion = euler2quat(z=orient_add, y=0, x=0)
            iquaternion = inverse_quaternion(quaternion)
            curr_orient = state[1:5]
            curr_transvel = state[15:18]

            new_orient = quaternion_product(iquaternion, curr_orient)
            if new_orient[0] < 0:
                new_orient = -new_orient
            new_translationalVelocity = rotate_by_quaternion(curr_transvel, iquaternion)
            state[1:5] = torch.FloatTensor(new_orient)
            state[15:18] = torch.FloatTensor(new_translationalVelocity)

            # Get action
            action = policy(state, True)
            action = action.data.numpy()
            state, reward, done, _ = cassie_env.step(action)
            state = torch.Tensor(state)
            if cassie_env.sim.qpos()[2] < 0.4:
                passed = 0
            count += 1
        if passed:
            save_data[j, 0] = passed
            save_data[j, 1] = -1
        else:
            save_data[j, :] = np.array([passed, count//(num_steps//2), cassie_env.speed, orient_add,\
                        cassie_env.speed-speed_schedule[max(0, speed_ind-2)], orient_schedule[orient_ind-1]])
            # if save_data[j, 1] == 0:
                # print("speed diff: ", speed_schedule[speed_ind-1]-speed_schedule[speed_ind-2])
                # print("curr speed: ", cassie_env.speed)
                # print("speed schedule: ", speed_schedule)
                # print("speed ind: ", speed_ind)
                # print("curr schedule: ", speed_schedule[speed_ind-1])
    return save_data, time.time() - start_t

# TODO: Change to create workers, then pass a single iter to each one. This way, in case a worker finishes before the others
# it can start running more iters. Can also add running stats of how many more tests to run, w/ loading bar
def eval_commands_multi_old(env_fn, policy, num_steps=200, num_commands=4, max_speed=3, min_speed=0, num_iters=4, num_procs=4, filename="test_eval_command.npy"):
    start_t1 = time.time()
    ray.init(num_cpus=num_procs)
    result_ids = []
    for i in range(num_procs):
        curr_iters = num_iters // num_procs
        if i == num_procs - 1:  # is last process to get launched, do remaining iters if not evenly divided between procs
            curr_iters = num_iters - i*curr_iters
        print("curr iters: ", curr_iters)
        args = (env_fn, policy, num_steps, num_commands, max_speed, min_speed, curr_iters)
        print("Starting worker ", i)
        result_ids.append(eval_commands_worker.remote(*args))
    result = ray.get(result_ids)
    # print(result)
    print("Got all results")
    total_data = np.concatenate([result[i][0] for i in range(num_procs)], axis=0)
    # print("timings: ", [result[i][1] for i in range(num_procs)])
    # print("sim timings: ", [result[i][2] for i in range(num_procs)])
    # # max_force = np.concatenate(result, axis=1)
    # print("total_data: ", total_data)
    np.save(filename, total_data)
    print("total time: ", time.time() - start_t1)
    ray.shutdown()