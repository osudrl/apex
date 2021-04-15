"""Python file for automatically running experiments from command line."""
import argparse

from apex import print_logo

from rl.envs.wrappers import SymmetricEnv

from rl.envs import Vectorize
from rl.envs.normalize import get_normalization_params, PreNormalizer

import functools

import torch

import numpy as np
import os
import time

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from colorama import Fore, Style

def make_env_fn(state_est=False):
    def _thunk():
        return CassieEnv("walking", clock_based=True, state_est=state_est)
    return _thunk


def gym_factory(path, **kwargs):
    from functools import partial

    """
    This is (mostly) equivalent to gym.make(), but it returns an *uninstantiated* 
    environment constructor.

    Since environments containing cpointers (e.g. Mujoco envs) can't be serialized, 
    this allows us to pass their constructors to Ray remote functions instead 
    (since the gym registry isn't shared across ray subprocesses we can't simply 
    pass gym.make() either)

    Note: env.unwrapped.spec is never set, if that matters for some reason.
    """
    spec = gym.envs.registry.spec(path)
    _kwargs = spec._kwargs.copy()
    _kwargs.update(kwargs)

    if callable(spec._entry_point):
        cls = spec._entry_point(**_kwargs)
    else:
        cls = gym.envs.registration.load(spec._entry_point)

    return partial(cls, **_kwargs)


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--redis_address", type=str, default=None)                  # address of redis server (for cluster setups)
parser.add_argument("--seed", type=int, default=1,help="RNG seed")
parser.add_argument("--logdir", type=str, default="./logs/ppo/experiments/", help="Where tensorboard should og diagnostics to")    
parser.add_argument("--name", type=str, default="model")
parser.add_argument("--state_est", type=bool, default=True)
parser.add_argument("--clock_based", default=True, action='store_true')
parser.add_argument("--mirror", default=False, action='store_true', help="Whether to use mirror environment or not")
parser.add_argument("--bounded",   type=bool, default=False)
parser.add_argument("--viz_port", default=8097, help="Visdom logging server port number")
# PPO algo args
parser.add_argument("--input_norm_steps", type=int, default=10000)
parser.add_argument("--n_itr", type=int, default=10000, help="Number of iterations of the learning algorithm")
parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate") # Xie
parser.add_argument("--eps", type=float, default=1e-5, help="Adam epsilon (for numerical stability)")
parser.add_argument("--lam", type=float, default=0.95, help="Generalized advantage estimate discount")
parser.add_argument("--gamma", type=float, default=0.99, help="MDP discount")
parser.add_argument("--entropy_coeff", type=float, default=0.0, help="Coefficient for entropy regularization")
parser.add_argument("--clip", type=float, default=0.2, help="Clipping parameter for PPO surrogate loss")
parser.add_argument("--minibatch_size", type=int, default=64, help="Batch size for PPO updates")
parser.add_argument("--epochs", type=int, default=3, help="Number of optimization epochs per PPO update") #Xie
parser.add_argument("--num_steps", type=int, default=5096, help="Number of sampled timesteps per gradient estimate")
parser.add_argument("--use_gae", type=bool, default=True,help="Whether or not to calculate returns using Generalized Advantage Estimation")
parser.add_argument("--num_procs", type=int, default=30, help="Number of threads to train on")
parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Value to clip gradients at.")
parser.add_argument("--max_traj_len", type=int, default=400, help="Max episode horizon")


args = parser.parse_args()
args.simrate = 50

args.num_procs = 56 #- 16*int(np.log2(60 / simrate))
args.num_steps = 12000 #// args.num_procs
args.input_norm_steps = 10000
args.minibatch_size = 2048
args.lr = 1e-4
args.epochs = 10
args.max_traj_len = 300# * int(60/args.simrate)
# args.seed = int(time.time())
args.max_grad_norm = 0.05
args.n_itr = 20000
args.use_gae = False
args.previous = None
args.learn_stddev = False
args.std_dev = -2.0
args.entropy_coeff = 0.0
args.anneal = 1.0#0.9995
# args.run_name = "noaccel_footdist_omniscient_speed0-3_freq1-2_footorient_seed{}".format(args.seed)
# args.logdir = "./logs/omniscient/"
# args.env_name = "CassieNoaccelFootDistOmniscient"
# args.run_name = "noaccel_footdist_fromscratch_step_smooth_pelheight_stddev-15_seed{}".format(args.seed)
# args.run_name = "noaccel_footdist_fromspeedmatch_speed0-3_freq1-2_footheightsmooth_footorient_height015_clockinputfix_stddev-2_epoch10_seed{}".format(args.seed)
# args.run_name = "noaccel_footdist_fromscratch_speed0-3_freq1-2_footheightsmooth_footorient_height015_clockinputfix_stddev-15_cont_stddev-2_cont_noDR_seed{}".format(args.seed)
# args.run_name = "fromnoreffull_speed0-3_freq1-2_norefrew_h015_hiprollvelact_rollgain80_yawgain80_stablepel_seed{}".format(args.seed)
# args.run_name = "fromscratch_speed0-3_freq1-2_norefrew_h015_rollyawgain80_rollyawphasetorque_rewterm04_seed{}".format(args.seed)
# args.run_name = "fromscratch_speed0-3_freq1-2_norefrew_h015_rollyawgain80_rollyawvelact_rewterm04_cont_rewterm03_cont_speedorientchange3_seed{}".format(args.seed)
# args.run_name = "fromscratch_speed0-4_footvarclock_linclock3_h030_cycletime1_mirror4_rewterm04_cont_rewterm03_seed{}".format(args.seed)
# args.run_name = "delta_fromscratch_speed0-1_freq1_h015_rollyawgain80_rollyawvelact_rewterm04_seed{}".format(args.seed)
# args.run_name = "cassie_noinertia_stddev25_rewterm04_cont_rewterm03_seed{}".format(args.seed)

# args.logdir = "./logs/footdist/"
# args.logdir = "./logs/sim2real/"
# args.env_name = "CassieNoaccelFootDist"

# args.run_name = "cassietray_speed-05-10_boxheightcost_norandvel_stddev15_rewterm04_cont_stddev20_rewterm03_seed{}".format(args.seed)
# args.logdir = "./logs/loadmass"
# args.env_name = "CassieClean"
# args.env_name = "CassieClean_pole"
# args.env_name = "CassieClean_tray"

args.run_name = "stand_fromwalk_switchreward_fheight20_stepinplace_half_trajlen300_cont_smooth_cont_trainpush_seed{}".format(args.seed)
# args.run_name = "fromrun_turnhalf10_40_speed0_quarter_seed{}".format(args.seed)
# args.run_name = "sprint_fromrun_trajlen300_stddev15_rewterm04_cont_purespeed_stddev20_rewterm03_cont_dynrand_seed{}".format(args.seed)
args.logdir = "./logs/extra"
# args.env_name = "CassieClean"
args.env_name = "CassieNoaccelFootDist"

# args.run_name = "mininput_fromscratch_speed0-3_freq1-2_norefrew_h015_rollyawgain80_rollyawvelact_rewterm04_learngains_cont_rewterm03_cont_seed{}".format(args.seed)
# args.logdir = "./logs/test/"
# args.env_name = "CassieNovelFootDist"
# args.env_name = "CassieMinInput"

# args.run_name = "nodelta_speedmatch_nomirror_speed0-1_fromconverge_seed{}".format(args.seed)
# args.env_name = "Cassie-v0"
# args.logdir = "./logs/thesis/"
# args.epochs = 3
# args.max_traj_len = 300

# args.run_name = "nomirror_seed{}".format(args.seed)
# args.logdir = "./logs/test/"
# args.run_name = "test"
# args.logdir = "./logs/dump"
# args.seed = 10
# args.input_norm_steps = 100
args.recurrent = False
args.algo_name = "ppo"
args.traj = "walking"
args.mirror = True
args.no_delta = True
args.state_est = True
args.dyn_random = False
args.clock_based = True
args.ik_baseline = False
args.learn_gains = False
args.phase_based = False
args.history = 0
# args.reward = "iros_paper"
# args.reward = "stand_up_pole_free_reward"
args.reward = "stand_walk_switch_smooth_reward"
# args.reward = "new_reward"
# args.reward = "tray_box_reward_easy"
# args.reward = "roll_over_reward"
# args.reward = "sprint_pure_speed_reward"
# args.reward = "speedmatch_footheightsmoothforce_footorient_hiprollyawvel_smoothact_torquecost_reward"
# args.reward = "speedmatchavg_footvarclock_footorient_stablepel_hiprollyawvel_smoothact_torquecost_reward"
# args.reward = "speedmatchavg_forcevel_footpos_footorient_stablepel_hiprollyawvel_smoothact_torquecost_reward"
# args.reward = "speedmatchavg_orientchange_forcevel_footpos_footorient_stablepel_hiprollyawvel_smoothact_torquecost_reward"
# args.reward = "speedmatchavg_forcevel_footpos_footorient_stablepel_hiprollyawvel_smoothact_torquecost_traybox_reward"
# prev_dir = "./logs/running/CassieNoaccelFootDist/"
# prev_dir = "./logs/test/CassieMinInput/"
# prev_dir = "./logs/sim2real/CassieNoaccelFootDist"
# prev_dir = "./logs/loadmass/CassieClean_tray/"
# prev_dir = "./logs/extra/CassieClean/"
prev_dir = "./logs/extra/CassieNoaccelFootDist/"

# args.previous = os.path.join(prev_dir, "sprint_fromrun_trajlen300_stddev15_rewterm04_cont_purespeed_stddev20_rewterm03_seed{}".format(args.seed))
# args.previous = os.path.join(prev_dir, "cassiepolex_stand_pole_free_angterm_pi2_stddev15_rewterm04_seed{}".format(args.seed))
# args.previous = os.path.join(prev_dir, "cassietray_speed-05-10_boxheightcost_norandvel_stddev15_rewterm04_seed{}".format(args.seed))
# args.previous = os.path.join(prev_dir, "run_turn10_25_stddev15_rewterm04_cont_stddev20_turnhalf10_40_cont_speed0_quarter_seed{}").format(args.seed)
args.previous = os.path.join(prev_dir, "stand_fromwalk_switchreward_fheight20_stepinplace_half_trajlen300_cont_smooth_seed{}").format(args.seed)
# args.previous = os.path.join(prev_dir, "run_speed0-4_forcevel02_footpos07_linclock5_h010-030_cycletime1_phaseadd15_mirror4_stddev15_rewterm04_cont_stddev20_rewterm03_seed{}".format(args.seed))

# prev_dir = "./logs/sim2real/CassieNoaccelFootDist/"
# args.previous = os.path.join(prev_dir, "cassie_notorquedelay_stddev15_rewterm04_seed{}".format(args.seed))

# prev_dir = "./logs/thesis/Cassie-v0"
# args.previous = os.path.join(prev_dir, "nodelta_trajmatch_nomirror_speed0-1_seed10".format(args.seed))

# Check if policy name already exists. If it does, increment filename
index = ''
while os.path.exists(os.path.join(args.logdir, args.env_name, args.run_name + index)):
    if index:
        index = '_(' + str(int(index[2:-1]) + 1) + ')'
    else:
        index = '_(1)'
args.run_name += index


if __name__ == "__main__":
    torch.set_num_threads(1) # see: https://github.com/pytorch/pytorch/issues/13757

    print_logo(subtitle="Distributed Proximal Policy Optimization")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # policy = GaussianMLP(
    #     obs_dim, action_dim, 
    #     nonlinearity="relu", 
    #     bounded=True, 
    #     init_std=np.exp(-2), 
    #     learn_std=False,
    #     normc_init=False
    # )
    # policy = torch.load("./trained_models/fwrd_walk_StateEst_speed-05-1_freq1_foottraj_land1.0.pt")
    # policy = torch.load("./trained_models/sidestep_StateEst_speedmatch_(3).pt")
    # policy = torch.load("./trained_models/sidestep_StateEst_speedmatch_footytraj_doublestance_time0.4_land0.4_vels_avgdiff_simrate15_bigweight_actpenalty_retrain.pt")

    # policy.obs_mean, policy.obs_std = map(torch.Tensor, get_normalization_params(iter=args.input_norm_steps, noise_std=2, policy=policy, env_fn=env_fn))
    # normalizer = PreNormalizer(iter=args.input_norm_steps, noise_std=2, policy=policy, online=False)
    # env = normalizer(Vectorize([env_fn]))
    # mean, std = env.ob_rms.mean, np.sqrt(env.ob_rms.var + 1E-8)
    # policy.obs_mean = torch.Tensor(mean)
    # policy.obs_std = torch.Tensor(std)

    # Load previous policy
    #policy = torch.load("./trained_models/sidestep/sidestep_StateEst_speed-05-2_freq1-2.pt")
    #policy.bounded = False

    from rl.algos.ppo import run_experiment
    run_experiment(args)

