import numpy as np
import torch
import time
import math
import ray
import sys, copy, os, copy

@ray.remote
class model_test_worker(object):
    def __init__ (self, id_num, env_fn, policy):
        self.id_num = id_num
        self.cassie_env = env_fn()
        self.policy = copy.deepcopy(policy)
        self.default_damping = self.cassie_env.sim.get_dof_damping()
        self.default_mass = self.cassie_env.sim.get_body_mass()
        self.default_fric = np.array([1, 0.005, 0.0001])
        torch.set_num_threads(1)

    @torch.no_grad()
    def test_model(self, damp, mass, fric):
        self.cassie_env.sim.set_dof_damping(damp)
        self.cassie_env.sim.set_body_mass(mass)
        self.cassie_env.sim.set_geom_friction(fric, "floor")
        state = self.cassie_env.reset_for_test()
        self.cassie_env.update_speed(0.5)
        for i in range(200):
            action = self.policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
            state = self.cassie_env.step_basic(action)
            if self.cassie_env.sim.qpos()[2] < 0.4:  # Failed, done testing
                return self.id_num, False, damp, mass, fric
        # print("eval time: ", time.time()-start_t)
        return self.id_num, True, damp, mass, fric

    @torch.no_grad()
    def test_model_indiv(self, test_type, ind, scale):
        if test_type == "mass":
            set_mass = copy.deepcopy(self.default_mass)
            set_mass[ind] *= scale
            self.cassie_env.sim.set_body_mass(set_mass)
            self.cassie_env.sim.set_dof_damping(self.default_damping)
            self.cassie_env.sim.set_geom_friction(self.default_fric, "floor")
        elif test_type == "damp":
            set_damp = copy.deepcopy(self.default_damping)
            set_damp[ind] *= scale
            self.cassie_env.sim.set_dof_damping(set_damp)
            self.cassie_env.sim.set_body_mass(self.default_mass)
            self.cassie_env.sim.set_geom_friction(self.default_fric, "floor")
        elif test_type == "fric":
            set_fric = copy.deepcopy(self.default_fric)
            set_fric *= scale
            self.cassie_env.sim.set_geom_friction(set_fric, "floor")
            self.cassie_env.sim.set_body_mass(self.default_mass)
            self.cassie_env.sim.set_dof_damping(self.default_damping)
        else:
            print("Error: Invalid test type.")
            exit()
        state = self.cassie_env.reset_for_test()
        self.cassie_env.update_speed(0.5)
        for i in range(200):
            action = self.policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
            state = self.cassie_env.step_basic(action)
            if self.cassie_env.sim.qpos()[2] < 0.4:  # Failed, done testing
                return self.id_num, False, test_type, ind, scale
        # print("eval time: ", time.time()-start_t)
        return self.id_num, True, test_type, ind, scale



@torch.no_grad()
def sensitivity_sweep(cassie_env, policy, factor):
    # Pelvis: 0->5
    # Hips: 6->8 and 19->21
    # Achilles: 9->11 and 22->24
    # Knees: 12 and 25
    # Tarsus: 14 and 27
    #
    # Total number of parameters: 17

    #parameter_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 19, 20, 21, 9, 10, 11, 22, 23,
    #        24, 12, 25, 14, 27]

    default_damp = cassie_env.default_damping
    parameter_ids = [(0, 5), (6, 8), (19, 21), (9, 11), (22, 24), (12), (25),
            (14), (27)]

    count = np.zeros(len(parameter_ids))
    for i in range(9):
        damp_range = np.copy(default_damp)
        if type(parameter_ids[i]) is tuple:
            for j in range(parameter_ids[i][0], parameter_ids[i][1]+1):
                # Set damp sweep
                damp_range[j] = default_damp[j] * factor
        else:
            damp_id = parameter_ids[i]
            damp_range[damp_id] = default_damp[damp_id] * factor

    
        state = torch.Tensor(cassie_env.full_reset()) 
        cassie_env.sim.set_dof_damping(np.clip(damp_range, 0, None))
        cassie_env.speed = 1
        cassie_env.side_speed = 0
        cassie_env.phase_add = 1
        
        curr_time = time.time()
        curr_time = cassie_env.sim.time()
        start_t = curr_time
        while curr_time < start_t + 15:
            action = policy(state, True)
            action = action.data.numpy()
            state, reward, done, _ = cassie_env.step(action)
            state = torch.Tensor(state)
            curr_time = cassie_env.sim.time()
            if cassie_env.sim.qpos()[2] < 0.4:
                count[i] = 1
                break

    return count

@torch.no_grad()
def eval_sensitivity(cassie_env, policy, incr, hi_factor, lo_factor):
    # this is dumb
    lo = 1.0
    lo_cnt = 0
    while lo >= lo_factor:
        lo -= incr
        lo_cnt += 1

    num_iters = int(hi_factor / incr) + lo_cnt + 1

    counter = 0

    # Matrix with the num_iters rows, and 9 + 1 columns. the first column is
    # the value of damping. the next nine indicate the parameter, 1 is a
    # failure at the value, 0 means either no failure or default val.
    ret = np.zeros((num_iters, 10))

    # Run the highs

    hi = 1.0

    while hi <= hi_factor:
        vals = sensitivity_sweep(cassie_env, policy, hi)
        ret[counter][0] = hi
        ret[counter][1:] = vals
        hi += incr
        counter += 1

    lo = 1.0

    # Run lo's
    for _ in range(lo_cnt):
        vals = sensitivity_sweep(cassie_env, policy, lo)
        ret[counter][0] = lo
        ret[counter][1:] = vals
        lo -= incr
        counter += 1

    # Report
    return ret


def make_model_args(default_damp, default_mass, damp_weak, damp_strong, mass_weak, mass_strong, fric_weak, fric_strong, num):
    default_fric = np.array([1, 0.005, 0.0001])

    hip_damps = np.linspace(default_damp[6:9]*damp_weak, default_damp[6:9]*damp_strong, num=num)
    achilles_damps = np.linspace(default_damp[9:12]*damp_weak, default_damp[9:12]*damp_strong, num=num)
    knee_damps = np.linspace(default_damp[12]*damp_weak, default_damp[12]*damp_strong, num=num)
    shin_damps = np.linspace(default_damp[13]*damp_weak, default_damp[13]*damp_strong, num=num)
    fcrank_damps = np.linspace(default_damp[16]*damp_weak, default_damp[16]*damp_strong, num=num)
    foot_damps = np.linspace(default_damp[18]*damp_weak, default_damp[18]*damp_strong, num=num)

    # side_damps = [np.hstack((hip_damps[i], achilles_damps[j], knee_damps[k], shin_damps[l], default_damp[14:16], fcrank_damps[m], 0, foot_damps[n]))
    #             for i in range(num) for j in range(num) for k in range(num) for l in range(num) for m in range(num) for n in range(num)]
    # all_damps = [np.concatenate((np.zeros(6), side_damps[i], side_damps[i])) for i in range(len(side_damps))]

    # Hip damp
    side_damps = [np.hstack((hip_damps[i], default_damp[9:19])) for i in range(num)]
    all_damps = [np.concatenate((np.zeros(6), side_damps[i], side_damps[i])) for i in range(num)]
    # Achilles damp
    side_damps = [np.hstack((default_damp[6:9], achilles_damps[i], default_damp[12:19])) for i in range(num)]
    all_damps += [np.concatenate((np.zeros(6), side_damps[i], side_damps[i])) for i in range(num)]
    # Knee damp
    side_damps = [np.hstack((default_damp[6:12], knee_damps[i], default_damp[13:19])) for i in range(num)]
    all_damps += [np.concatenate((np.zeros(6), side_damps[i], side_damps[i])) for i in range(num)]
    # Shine damp
    side_damps = [np.hstack((default_damp[6:13], shin_damps[i], default_damp[14:19])) for i in range(num)]
    all_damps += [np.concatenate((np.zeros(6), side_damps[i], side_damps[i])) for i in range(num)]
    # fcrank damp
    side_damps = [np.hstack((default_damp[6:16], fcrank_damps[i], default_damp[17:19])) for i in range(num)]
    all_damps += [np.concatenate((np.zeros(6), side_damps[i], side_damps[i])) for i in range(num)]
    # foot damp
    side_damps = [np.hstack((default_damp[6:18], foot_damps[i])) for i in range(num)]
    all_damps += [np.concatenate((np.zeros(6), side_damps[i], side_damps[i])) for i in range(num)]

    pel_mass = np.linspace(default_mass[1]*mass_weak, default_mass[1]*mass_strong, num=num)
    hip_mass = np.linspace(default_mass[2:5]*mass_weak, default_mass[2:5]*mass_strong, num=num)
    achilles_mass = np.linspace(default_mass[5]*mass_weak, default_mass[5]*mass_strong, num=num)
    knee_mass = np.linspace(default_mass[6]*mass_weak, default_mass[6]*mass_strong, num=num)
    knee_spring_mass = np.linspace(default_mass[7]*mass_weak, default_mass[7]*mass_strong, num=num)
    shin_mass = np.linspace(default_mass[8]*mass_weak, default_mass[8]*mass_strong, num=num)
    tarsus_mass = np.linspace(default_mass[9]*mass_weak, default_mass[9]*mass_strong, num=num)
    heel_spring_mass = np.linspace(default_mass[10]*mass_weak, default_mass[10]*mass_strong, num=num)
    fcrank_mass = np.linspace(default_mass[11]*mass_weak, default_mass[11]*mass_strong, num=num)
    prod_mass = np.linspace(default_mass[12]*mass_weak, default_mass[12]*mass_strong, num=num)
    foot_mass = np.linspace(default_mass[13]*mass_weak, default_mass[13]*mass_strong, num=num)
    default_side_mass = default_mass[2:14]

    # all_mass = [np.hstack((0, pel_mass[i], default_mass[2:13], foot_mass[j], default_mass[2:13], foot_mass[j]))
    #             for i in range(num) for j in range(num)]

    change_mass = [achilles_mass, knee_mass, knee_spring_mass, shin_mass, tarsus_mass, heel_spring_mass, fcrank_mass, prod_mass, foot_mass]
    # before_inds = [(0, 0), (2, 5), (2, 6), (2, 7), (2, 8)]
    # after_inds = [(5:13)]
    all_mass = [np.hstack((0, pel_mass[i], default_mass[2:])) for i in range(num)]
    # side_mass = [np.hstack((hip_mass[i], default_mass[5:13])) for i in range(num)]
    side_mass = [default_side_mass] * num
    for i in range(num):
        side_mass[i][0:3] = hip_mass[i]
    all_mass += [np.hstack((0, default_mass[1], side_mass[i], side_mass[i])) for i in range(num)]
    for i in range(len(change_mass)):
        # side_mass = [np.hstack((default_mass[2:5+i], change_mass[i][j], default_mass[6+i:after_inds[i][1]]))
                        # for j in range(num)]
        side_mass = [default_side_mass] * num
        for j in range(num):
            side_mass[j][3+i] = change_mass[i][j]
        all_mass += [np.hstack((0, default_mass[1], side_mass[j], side_mass[j])) for j in range(num)]

    frictions = np.linspace(default_fric*fric_weak, default_fric*fric_strong, num=num)

    # args = [(damp, mass, fric) for damp in all_damps for mass in all_mass for fric in frictions]
    args = [(damp, default_mass, default_fric) for damp in all_damps]
    args += [(default_damp, mass, default_fric) for mass in all_mass]
    args += [(default_damp, default_mass, fric) for fric in frictions]

    return args

def make_model_args2(damp_weak, damp_strong, mass_weak, mass_strong, fric_weak, fric_strong, num):
    # Damping
    inds = [[6,7,8,19,20,21], [9,10,11,22,23,24], [12,25], [13,26], [16,29], [18,31]]
    scales = np.linspace(damp_weak, damp_strong, num=num)
    args = [("damp", ind, scale) for ind in inds for scale in scales]

    # Mass
    inds = [1, [2,3,4,14,15,16], [5,17], [6,18], [7,19], [8,20], [9,21], [10,22], [11,23], [12,24], [13,25]]
    scales = np.linspace(mass_weak, mass_strong, num=num)
    args += [("mass", ind, scale) for ind in inds for scale in scales]

    # Frictions
    scales = np.linspace(fric_weak, fric_strong, num=num)
    args += [("fric", 0, scale) for scale in scales]

    return args



def eval_model_multi(env_fn, policy, weak, strong, num, num_procs):
    cassie_env = env_fn()
    default_damping = cassie_env.sim.get_dof_damping()
    default_mass = cassie_env.sim.get_body_mass()
    test_args = make_model_args(default_damping, default_mass, weak, strong, weak, strong, weak, strong, num)
    print("num args: ", len(test_args))
    # exit()

    # Make and start all workers
    print("Using {} processes".format(num_procs))
    ray.shutdown()
    ray.init(num_cpus=num_procs)
    workers = [model_test_worker.remote(i, env_fn, policy) for i in range(num_procs)]
    print("made workers")
    result_ids = [workers[i].test_model.remote(*test_args[i]) for i in range(num_procs)]
    print("started workers")
    curr_arg_ind = num_procs

    num_args = len(test_args) 
    pass_data = [0]*num_args
    damp_data = [0]*num_args
    mass_data = [0]*num_args
    fric_data = [0]*num_args
    arg_count = 0
    sys.stdout.write("Finished {} out of {} tests".format(arg_count, num_args))
    sys.stdout.flush()
    start_t = time.time()
    while result_ids:
        done_id = ray.wait(result_ids, num_returns=1, timeout=None)[0][0]
        worker_id, success, damp, mass, fric = ray.get(done_id)
        pass_data[arg_count] = success
        damp_data[arg_count] = damp
        mass_data[arg_count] = mass
        fric_data[arg_count] = fric

        result_ids.remove(done_id)
        if curr_arg_ind < num_args:
            result_ids.append(workers[worker_id].test_model.remote(*test_args[curr_arg_ind]))
        curr_arg_ind += 1
        arg_count += 1
        elapsed_time = time.time() - start_t
        time_left = elapsed_time/arg_count * (num_args-arg_count)
        sys.stdout.write("\rFinished {} out of {} tests. {:.1f}s elapsed, {:.1f}s left".format(arg_count, num_args, elapsed_time, time_left))
        sys.stdout.flush()
        # TODO: Add progress bar and estimated time left
    print()
    print("Total time: ", time.time() - start_t)
    ray.shutdown()
    
    return pass_data, damp_data, mass_data, fric_data

def eval_model_multi2(env_fn, policy, weak, strong, num, num_procs):
    cassie_env = env_fn()
    default_damping = cassie_env.sim.get_dof_damping()
    default_mass = cassie_env.sim.get_body_mass()
    default_fric = np.array([1, 0.005, 0.0001])
    # test_args = make_model_args(default_damping, default_mass, weak, strong, weak, strong, weak, strong, num)
    test_args = make_model_args2(.1, 5, weak, strong, weak, strong, num)
    print("num args: ", len(test_args))
    # exit()

    # Make and start all workers
    print("Using {} processes".format(num_procs))
    ray.shutdown()
    ray.init(num_cpus=num_procs)
    workers = [model_test_worker.remote(i, env_fn, policy) for i in range(num_procs)]
    print("made workers")
    result_ids = [workers[i].test_model_indiv.remote(*test_args[i]) for i in range(num_procs)]
    print("started workers")
    curr_arg_ind = num_procs

    num_args = len(test_args) 
    pass_data = [0]*num_args
    type_data = [0]*num_args
    scale_data = [0]*num_args
    arg_count = 0
    sys.stdout.write("Finished {} out of {} tests".format(arg_count, num_args))
    sys.stdout.flush()
    start_t = time.time()
    while result_ids:
        done_id = ray.wait(result_ids, num_returns=1, timeout=None)[0][0]
        worker_id, success, test_type, ind, scale = ray.get(done_id)
        pass_data[arg_count] = success
        type_data[arg_count] = ind
        scale_data[arg_count] = scale

        result_ids.remove(done_id)
        if curr_arg_ind < num_args:
            result_ids.append(workers[worker_id].test_model_indiv.remote(*test_args[curr_arg_ind]))
        curr_arg_ind += 1
        arg_count += 1
        elapsed_time = time.time() - start_t
        time_left = elapsed_time/arg_count * (num_args-arg_count)
        sys.stdout.write("\rFinished {} out of {} tests. {:.1f}s elapsed, {:.1f}s left".format(arg_count, num_args, elapsed_time, time_left))
        sys.stdout.flush()
        # TODO: Add progress bar and estimated time left
    print()
    print("Total time: ", time.time() - start_t)
    ray.shutdown()
    
    return pass_data, type_data, scale_data

def get_model_stats(path):
    data = np.load(os.path.join(path, "./eval_model.npz"))
    print(data.files)
    pass_data = data["pass_data"]
    print(np.sum(pass_data)/len(pass_data))