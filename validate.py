#validate the sensitivity of parameters

##########To do#########
#Formalize desired output
#Get to run with user input of env and policy without iteration
#Iterate through each parameter

def iterativeValidation(env, policy):

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