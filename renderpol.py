from rl.utils import renderpolicy, rendermultipolicy, renderpolicy_speedinput, rendermultipolicy_speedinput
from cassie import CassieEnv
# from rl.policies import GaussianMLP, BetaMLP

# from cassie.slipik_env import CassieIKEnv
# from cassie.no_delta_env import CassieEnv_nodelta
# from cassie.speed_env import CassieEnv_speed
# from cassie.speed_double_freq_env import CassieEnv_speed_dfreq
# from cassie.speed_no_delta_env import CassieEnv_speed_no_delta
# from cassie.speed_no_delta_neutral_foot_env import CassieEnv_speed_no_delta_neutral_foot
# from cassie.standing_env import CassieEnv_stand
# from cassie.speed_sidestep_env import CassieEnv_speed_sidestep
from cassie.aslipik_unified_env import UnifiedCassieIKEnv
from cassie.aslipik_unified_env_alt_reward import UnifiedCassieIKEnvAltReward
from cassie.aslipik_unified_env_task_reward import UnifiedCassieIKEnvTaskReward
from cassie.aslipik_unified_no_delta_env import UnifiedCassieIKEnvNoDelta

import torch

import numpy as np
import os
import time

# cassie_env = CassieEnv("walking", clock_based=True, state_est=True)
# cassie_env = CassieEnv_nodelta("walking", clock_based=True, state_est=False)
# cassie_env = CassieEnv_speed("walking", clock_based=True, state_est=True)
# cassie_env = CassieEnv_speed_dfreq("walking", clock_based=True, state_est=False)
# cassie_env = CassieEnv_speed_no_delta("walking", clock_based=True, state_est=False)
# cassie_env = CassieEnv_speed_no_delta_neutral_foot("walking", clock_based=True, state_est=True)
# cassie_env = CassieEnv_speed_sidestep("walking", clock_based=True, state_est=True)
cassie_env = UnifiedCassieIKEnvNoDelta("walking", clock_based=True, state_est=True, debug=True)
# cassie_env = CassieEnv_stand(state_est=False)

# policy = torch.load("./trained_models/stiff_spring/stiff_StateEst_speed2.pt")
# policy = torch.load("./trained_models/sidestep_StateEst_footxypenaltysmall_forcepenalty_hipyaw_limittargs_pelaccel3_speed-05-1_side03_freq1.pt")
policy = torch.load("./trained_models/aslip_unified_no_delta_10_v6.pt")
# policy = torch.load("./trained_models/aslip_unified_no_delta_0_v4.pt")
policy.eval()
renderpolicy_speedinput(cassie_env, policy, deterministic=True, dt=0.05, speedup = 2)