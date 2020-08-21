# We use this directory for keeping track of reward functions. Each reward function operates on an object of CassieEnv_v2, passed as 'self'

from .clock_rewards import *
from .aslip_rewards import *
from .rnn_dyn_random_reward import *
from .iros_paper_reward import *
from .command_reward import *

# from .speedmatch_footorient_joint_smooth_reward import *
from .speedmatch_rewards import *
from .trajmatch_reward import *
from .standing_rewards import *
# from .speedmatch_heuristic_reward import *
from .side_speedmatch_rewards import *
# from .side_speedmatch_foottraj_reward import *
# from .side_speedmatch_heightvel_reward import *
# from .side_speedmatch_heuristic_reward import *
# from .side_speedmatch_torquesmooth_reward import *