# Unified
from .cassie import CassieEnv
from .cassie_traj import CassieTrajEnv
from .cassie_playground import CassiePlayground
from .cassie_standing_env import CassieStandingEnv  # sorta old/unused

# Proprietary
from .cassie_noaccel_footdist_omniscient import CassieEnv_noaccel_footdist_omniscient
from .cassie_footdist_env import CassieEnv_footdist
from .cassie_noaccel_footdist_env import CassieEnv_noaccel_footdist
from .cassie_noaccel_footdist_nojoint_env import CassieEnv_noaccel_footdist_nojoint
from .cassie_novel_footdist_env import CassieEnv_novel_footdist
from .cassie_mininput_env import CassieEnv_mininput

# CassieMujocoSim
from .cassiemujoco import *


##############
# DEPRECATED #
##############
# from .cassie_env import CassieEnv
# from .taskspace_env import CassieTSEnv
# from .aslipik_env import CassieIKEnv
# from .aslipik_unified_env import UnifiedCassieIKEnv
# from .aslipik_unified_no_delta_env import UnifiedCassieIKEnvNoDelta
# from .no_delta_env import CassieEnv_nodelta
# from .dynamics_random import CassieEnv_rand_dyn
# from .speed_double_freq_env import CassieEnv_speed_dfreq
# from .ground_friction_env import CassieGroundFrictionEnv
# from .cassie_standing_env import CassieStandingEnv
