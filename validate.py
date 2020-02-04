#validate the sensitivity of policy to environment parameters

##########To do#########
#Formalize desired output
#Get to run with user input of env and policy without iteration
#Iterate through each parameter

def iterativeValidation(policy, env, max_traj_len=1000, visualize=True, env_name=None, speed=0.0, state_est=True, clock_based=False):
    # Follow apex.py and use an env_factory, or expect an env on input?

    #if env_name is None:
    #    env = env_factory(policy.env_name, speed=speed, state_est=state_est, clock_based=clock_based)()
    #else:
    #    env = env_factory(env_name, speed=speed, state_est=state_est, clock_based=clock_based)()

    # This stuff was ripped out from Jonah's dynamics randomization.
    # We are leaving it as is while building the infrastructure for testing.
    # Future plan: Put all the damp/mass ranges into a tuple and iterate through,
    # at each iteration running a test with the current parameter randomized, and all
    # others default. Then log results and output in a sane and useful way.

    damp = env.default_damping
    weak_factor = 0.5
    strong_factor = 1.5
    pelvis_damp_range = [[damp[0], damp[0]], 
                        [damp[1], damp[1]], 
                        [damp[2], damp[2]], 
                        [damp[3], damp[3]], 
                        [damp[4], damp[4]], 
                        [damp[5], damp[5]]]                 # 0->5

    hip_damp_range = [[damp[6]*weak_factor, damp[6]*strong_factor],
                    [damp[7]*weak_factor, damp[7]*strong_factor],
                    [damp[8]*weak_factor, damp[8]*strong_factor]]  # 6->8 and 19->21

    achilles_damp_range = [[damp[9]*weak_factor,  damp[9]*strong_factor],
                            [damp[10]*weak_factor, damp[10]*strong_factor], 
                            [damp[11]*weak_factor, damp[11]*strong_factor]] # 9->11 and 22->24

    knee_damp_range     = [[damp[12]*weak_factor, damp[12]*strong_factor]]   # 12 and 25
    shin_damp_range     = [[damp[13]*weak_factor, damp[13]*strong_factor]]   # 13 and 26
    tarsus_damp_range   = [[damp[14], damp[14]]]             # 14 and 27
    heel_damp_range     = [[damp[15], damp[15]]]                           # 15 and 28
    fcrank_damp_range   = [[damp[16]*weak_factor, damp[16]*strong_factor]]   # 16 and 29
    prod_damp_range     = [[damp[17], damp[17]]]                           # 17 and 30
    foot_damp_range     = [[damp[18]*weak_factor, damp[18]*strong_factor]]   # 18 and 31

    side_damp = hip_damp_range + achilles_damp_range + knee_damp_range + shin_damp_range + tarsus_damp_range + heel_damp_range + fcrank_damp_range + prod_damp_range + foot_damp_range
    damp_range = pelvis_damp_range + side_damp + side_damp
    damp_noise = [np.random.uniform(a, b) for a, b in damp_range]

    hi = 1.3
    lo = 0.7
    m = env.default_mass
    pelvis_mass_range      = [[lo*m[1],  hi*m[1]]]  # 1
    hip_mass_range         = [[lo*m[2],  hi*m[2]],  # 2->4 and 14->16
                            [lo*m[3],  hi*m[3]], 
                            [lo*m[4],  hi*m[4]]] 

    achilles_mass_range    = [[lo*m[5],  hi*m[5]]]  # 5 and 17
    knee_mass_range        = [[lo*m[6],  hi*m[6]]]  # 6 and 18
    knee_spring_mass_range = [[lo*m[7],  hi*m[7]]]  # 7 and 19
    shin_mass_range        = [[lo*m[8],  hi*m[8]]]  # 8 and 20
    tarsus_mass_range      = [[lo*m[9],  hi*m[9]]]  # 9 and 21
    heel_spring_mass_range = [[lo*m[10], hi*m[10]]] # 10 and 22
    fcrank_mass_range      = [[lo*m[11], hi*m[11]]] # 11 and 23
    prod_mass_range        = [[lo*m[12], hi*m[12]]] # 12 and 24
    foot_mass_range        = [[lo*m[13], hi*m[13]]] # 13 and 25

    side_mass = hip_mass_range + achilles_mass_range \
                + knee_mass_range + knee_spring_mass_range \
                + shin_mass_range + tarsus_mass_range \
                + heel_spring_mass_range + fcrank_mass_range \
                + prod_mass_range + foot_mass_range

    mass_range = [[0, 0]] + pelvis_mass_range + side_mass + side_mass
    mass_noise = [np.random.uniform(a, b) for a, b in mass_range]

    delta_y_min, delta_y_max = env.default_ipos[4] - 0.07, env.default_ipos[4] + 0.07
    delta_z_min, delta_z_max = env.default_ipos[5] - 0.04, env.default_ipos[5] + 0.04
    com_noise = [0, 0, 0] + [np.random.uniform(-0.25, 0.06)] + [np.random.uniform(delta_y_min, delta_y_max)] + [np.random.uniform(delta_z_min, delta_z_max)] + list(env.default_ipos[6:])

    fric_noise = [np.random.uniform(0.4, 1.4)] + [np.random.uniform(3e-3, 8e-3)] + list(env.default_fric[2:])

    env.sim.set_dof_damping(np.clip(damp_noise, 0, None))
    env.sim.set_body_mass(np.clip(mass_noise, 0, None))
    env.sim.set_body_ipos(com_noise)
    env.sim.set_ground_friction(np.clip(fric_noise, 0, None))

    # From policy_eval
    while True:
        state = env.reset()
        done = False
        timesteps = 0
        eval_reward = 0
        while not done and timesteps < max_traj_len:

        if hasattr(env, 'simrate'):
            start = time.time()
        
        action = policy.forward(torch.Tensor(state)).detach().numpy()
        state, reward, done, _ = env.step(action)
        if visualize:
            env.render()
        eval_reward += reward
        timesteps += 1

        if hasattr(env, 'simrate'):
            # assume 30hz (hack)
            end = time.time()
            delaytime = max(0, 1000 / 30000 - (end-start))
            time.sleep(delaytime)

        print("Eval reward: ", eval_reward)



#nbody layout:
# 0:  worldbody (zero)
# 1:  pelvis

# 2:  left hip roll 
# 3:  left hip yaw
# 4:  left hip pitch
# 5:  left achilles rod
# 6:  left knee
# 7:  left knee spring
# 8:  left shin
# 9:  left tarsus
# 10: left heel spring
# 12: left foot crank
# 12: left plantar rod
# 13: left foot

# 14: right hip roll 
# 15: right hip yaw
# 16: right hip pitch
# 17: right achilles rod
# 18: right knee
# 19: right knee spring
# 20: right shin
# 21: right tarsus
# 22: right heel spring
# 23: right foot crank
# 24: right plantar rod
# 25: right foot


# qpos layout
# [ 0] Pelvis x
# [ 1] Pelvis y
# [ 2] Pelvis z
# [ 3] Pelvis orientation qw
# [ 4] Pelvis orientation qx
# [ 5] Pelvis orientation qy
# [ 6] Pelvis orientation qz
# [ 7] Left hip roll         (Motor [0])
# [ 8] Left hip yaw          (Motor [1])
# [ 9] Left hip pitch        (Motor [2])
# [10] Left achilles rod qw
# [11] Left achilles rod qx
# [12] Left achilles rod qy
# [13] Left achilles rod qz
# [14] Left knee             (Motor [3])
# [15] Left shin                        (Joint [0])
# [16] Left tarsus                      (Joint [1])
# [17] Left heel spring
# [18] Left foot crank
# [19] Left plantar rod
# [20] Left foot             (Motor [4], Joint [2])
# [21] Right hip roll        (Motor [5])
# [22] Right hip yaw         (Motor [6])
# [23] Right hip pitch       (Motor [7])
# [24] Right achilles rod qw
# [25] Right achilles rod qx
# [26] Right achilles rod qy
# [27] Right achilles rod qz
# [28] Right knee            (Motor [8])
# [29] Right shin                       (Joint [3])
# [30] Right tarsus                     (Joint [4])
# [31] Right heel spring
# [32] Right foot crank
# [33] Right plantar rod
# [34] Right foot            (Motor [9], Joint [5])

# qvel layout
# [ 0] Pelvis x
# [ 1] Pelvis y
# [ 2] Pelvis z
# [ 3] Pelvis orientation wx
# [ 4] Pelvis orientation wy
# [ 5] Pelvis orientation wz
# [ 6] Left hip roll         (Motor [0])
# [ 7] Left hip yaw          (Motor [1])
# [ 8] Left hip pitch        (Motor [2])
# [ 9] Left knee             (Motor [3])
# [10] Left shin                        (Joint [0])
# [11] Left tarsus                      (Joint [1])
# [12] Left foot             (Motor [4], Joint [2])
# [13] Right hip roll        (Motor [5])
# [14] Right hip yaw         (Motor [6])
# [15] Right hip pitch       (Motor [7])
# [16] Right knee            (Motor [8])
# [17] Right shin                       (Joint [3])
# [18] Right tarsus                     (Joint [4])
# [19] Right foot            (Motor [9], Joint [5])
