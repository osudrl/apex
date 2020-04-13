import numpy as np
import torch
import time
import math

#from eval_perturb import reset_to_phase

@torch.no_grad()
def eval_sensitivity(cassie_env, policy, incr):
    # Pelvis: 0->5
    # Hips: 6->8 and 19->21
    # Achilles: 9->11 and 22->24
    # Knees: 12 and 25
    # Tarsus: 14 and 27
    #
    # Total number of parameters: 17

    #parameter_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 19, 20, 21, 9, 10, 11, 22, 23,
    #        24, 12, 25, 14, 27]

    damp_value_lo = .1
    damp_value_hi = 90

    default_damp = cassie_env.default_damping
    parameter_ids = [(0, 5), (6, 8), (19, 21), (9, 11), (22, 24), (12), (25),
            (14), (27)]

    count = 0
    for i in range(9):
    
        damp_range_lo = np.copy(default_damp)
        damp_range_hi = np.copy(default_damp)
        if type(parameter_ids[i]) is tuple:
            for j in range(parameter_ids[i][0], parameter_ids[i][1]+1):
                # Set damp sweep
                damp_range_lo[j] = default_damp[j] * damp_value_lo
                damp_range_hi[j] = default_damp[j] * damp_value_hi
        else:
            damp_id = parameter_ids[i]
            damp_range_lo[damp_id] = default_damp[damp_id] * damp_value_lo
            damp_range_hi[damp_id] = default_damp[damp_id] * damp_value_hi


        for run in range(2):
            state = torch.Tensor(cassie_env.full_reset()) 
            if run == 0:
                cassie_env.sim.set_dof_damping(np.clip(damp_range_lo, 0, None))
            else:
                cassie_env.sim.set_dof_damping(np.clip(damp_range_hi, 0, None))
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
                    count += 1
                    break
    print("Failed", count, "times") 

