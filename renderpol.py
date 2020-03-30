from rl.utils import renderpolicy, rendermultipolicy, renderpolicy_speedinput, rendermultipolicy_speedinput
from cassie import CassieEnv
from rl.policies import GaussianMLP, BetaMLP
from cassie.slipik_env import CassieIKEnv
from cassie.no_delta_env import CassieEnv_nodelta
from cassie.speed_env import CassieEnv_speed
from cassie.speed_double_freq_env import CassieEnv_speed_dfreq
from cassie.speed_no_delta_env import CassieEnv_speed_no_delta
from cassie.speed_no_delta_neutral_foot_env import CassieEnv_speed_no_delta_neutral_foot
from cassie.standing_env import CassieEnv_stand
from cassie.speed_sidestep_env import CassieEnv_speed_sidestep
from cassie.speed_no_delta_noheight_noaccel_env import CassieEnv_speed_no_delta_noheight_noaccel
from cassie.quaternion_function import *

import torch

import numpy as np
import os
import time

# cassie_env = CassieEnv("walking", clock_based=True, state_est=True)
# cassie_env = CassieEnv_nodelta("walking", clock_based=True, state_est=False)
# cassie_env = CassieEnv_speed("walking", clock_based=True, state_est=True)
# cassie_env = CassieEnv_speed_dfreq("walking", clock_based=True, state_est=False)
# cassie_env = CassieEnv_speed_no_delta("walking", clock_based=True, state_est=False)
cassie_env = CassieEnv_speed_no_delta_neutral_foot("walking", simrate=60, clock_based=True, state_est=True)
# cassie_env = CassieEnv_speed_sidestep("walking", clock_based=True, state_est=True)
# cassie_env = CassieEnv_speed_no_delta_noheight_noaccel("walking", simrate=60, state_est=True)
# cassie_env = CassieEnv_stand(state_est=False)

# policy = torch.load("./trained_models/stiff_spring/stiff_StateEst_speed2.pt")
# policy = torch.load("./trained_models/sidestep_StateEst_speedmatch_actpenaltybig_footorient_footheightvel01.pt")
# policy = torch.load("./trained_models/noheightaccel_StateEst_speedmatch_randfric_mirror.pt")
# policy = torch.load("./trained_models/5k_footorient_actpenaltybig_footheightvellow_randjoint_randfric_seed10.pt")
policy = torch.load("./trained_models/5k_retrain_speed0-3_freq1-2_seed40.pt")
# policy = torch.load("./trained_models/5k_retrain_speed0-2_freq1-15_seed10.pt")
# policy = torch.load("./trained_models/fwrd_walk_StateEst_speed0-3_freq1-2.pt")
# policy = torch.load("./trained_models/nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2.pt")

policy.eval()
# policy.bounded = False
smallinput = False

# default_fric = cassie_env.sim.get_geom_friction()
# fric_noise = []
# for _ in range(int(len(default_fric)/3)):
#     fric_noise += [0.1, 1e-3, 1e-4]
# cassie_env.sim.set_geom_friction([0.25, 1e-6, 5e-5], "floor")
# rand_angle = np.pi/180*np.array([-5, -5])#np.random.uniform(-5, 5, 2)
# floor_quat = euler2quat(z=0, y=rand_angle[0], x=rand_angle[1])
# cassie_env.sim.set_geom_quat(floor_quat, "floor")
default_damping = cassie_env.sim.get_dof_damping()
weak_factor = .10
strong_factor = .10

pelvis_damp_range = [[default_damping[0], default_damping[0]], 
                    [default_damping[1], default_damping[1]], 
                    [default_damping[2], default_damping[2]], 
                    [default_damping[3], default_damping[3]], 
                    [default_damping[4], default_damping[4]], 
                    [default_damping[5], default_damping[5]]] 

hip_damp_range = [[default_damping[6]*weak_factor, default_damping[6]*strong_factor],
                [default_damping[7]*weak_factor, default_damping[7]*strong_factor],
                [default_damping[8]*weak_factor, default_damping[8]*strong_factor]]  # 6->8 and 19->21

achilles_damp_range = [[default_damping[9]*weak_factor,  default_damping[9]*strong_factor],
                        [default_damping[10]*weak_factor, default_damping[10]*strong_factor], 
                        [default_damping[11]*weak_factor, default_damping[11]*strong_factor]] # 9->11 and 22->24

knee_damp_range     = [[default_damping[12]*weak_factor, default_damping[12]*strong_factor]]   # 12 and 25
shin_damp_range     = [[default_damping[13]*weak_factor, default_damping[13]*strong_factor]]   # 13 and 26
tarsus_damp_range   = [[default_damping[14], default_damping[14]]]             # 14 and 27
heel_damp_range     = [[default_damping[15], default_damping[15]]]                           # 15 and 28
fcrank_damp_range   = [[default_damping[16]*weak_factor, default_damping[16]*strong_factor]]   # 16 and 29
prod_damp_range     = [[default_damping[17], default_damping[17]]]                           # 17 and 30
foot_damp_range     = [[default_damping[18]*weak_factor, default_damping[18]*strong_factor]]   # 18 and 31

side_damp = hip_damp_range + achilles_damp_range + knee_damp_range + shin_damp_range + tarsus_damp_range + heel_damp_range + fcrank_damp_range + prod_damp_range + foot_damp_range
damp_range = pelvis_damp_range + side_damp + side_damp
damp_noise = [np.random.uniform(a, b) for a, b in damp_range]
# cassie_env.sim.set_dof_damping(np.clip(damp_noise, 0, None))

renderpolicy_speedinput(cassie_env, policy, smallinput=smallinput, deterministic=True, dt=0.05, speedup = 5)

# seeds = [10, 20, 30, 40, 50]
# policies = []
# for seed in seeds:
#     policy = torch.load("./trained_models/5k_footorient_smoothcost_jointreward_randjoint_seed{}.pt".format(seed))
#     policy.eval()
#     policies.append(policy)


# rendermultipolicy_speedinput(cassie_env, policies, deterministic=True, dt=0.05, speedup = 3)

