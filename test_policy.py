from cassie import CassiePlayground
from tools.test_commands import *
from tools.eval_perturb import *
from tools.eval_mission import *
from tools.compare_pols import *
from tools.eval_sensitivity import *
from collections import OrderedDict
from util.env import env_factory

import torch
import pickle
import os, sys, argparse
import numpy as np

# Get policy to test from args, load policy and env
parser = argparse.ArgumentParser()
# General args
parser.add_argument("--path", type=str, default="./trained_models/nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2", help="path to folder containing policy and run details")
parser.add_argument("--path2", type=str, default="./trained_models/nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2", help="path to folder containing 2nd policy to compare against")
parser.add_argument("--n_procs", type=int, default=4, help="Number of procs to use for multi-processing")
parser.add_argument("--test", type=str, default="full", help="Test to run (options: \"full\", \"commands\", and \"perturb\", and \"compare\")")
parser.add_argument("--eval", default=True, action="store_false", help="Whether to call policy.eval() or not")
# Test Commands args
parser.add_argument("--n_steps", type=int, default=200, help="Number of steps to for a full command cycle (1 speed change and 1 orientation change)")
parser.add_argument("--n_commands", type=int, default=6, help="Number of commands in a single test iteration")
parser.add_argument("--max_speed", type=float, default=3.0, help="Maximum allowable speed to test")
parser.add_argument("--min_speed", type=float, default=0.0, help="Minimum allowable speed to test")
parser.add_argument("--n_iter", type=int, default=10000, help="Number of command cycles to test")
# Test Perturbs args
parser.add_argument("--wait_time", type=float, default=3.0, help="How long to wait after perturb to count as success")
parser.add_argument("--pert_dur", type=float, default=0.2, help="How long to apply perturbation")
parser.add_argument("--pert_size", type=float, default=50, help="Size of perturbation to start sweep from")
parser.add_argument("--pert_incr", type=float, default=10.0, help="How much to increment the perturbation size after each success")
parser.add_argument("--pert_body", type=str, default="cassie-pelvis", help="Body to apply perturbation to")
parser.add_argument("--num_angles", type=int, default=100, help="How many angles to test (angles are evenly divided into 2*pi)")
# Test Mission args
parser.add_argument("--viz", default=False, action='store_true')
# Test parameter sensitivity args
parser.add_argument("--sens_incr", type=float, default=0.05, help="Size of increments for the sensityivity sweep")
parser.add_argument("--hi_factor", type=float, default=15, help="High factor")
parser.add_argument("--lo_factor", type=float, default=0, help="Low factor")

args = parser.parse_args()
run_args = pickle.load(open(os.path.join(args.path, "experiment.pkl"), "rb"))
# cassie_env = CassieEnv(traj=run_args.traj, clock_based=run_args.clock_based, state_est=run_args.state_est, dynamics_randomization=run_args.dyn_random)
# env_fn = partial(CassieEnv, traj=run_args.traj, clock_based=run_args.clock_based, state_est=run_args.state_est, dynamics_randomization=run_args.dyn_random)
# Make mirror False so that env_factory returns a regular wrap env function and not a symmetric env function that can be called to return
# a cassie environment (symmetric env cannot be called to make another env)
env_fn = env = env_factory(
    run_args.env_name,
    command_profile=run_args.command_profile,
    input_profile=run_args.input_profile,
    simrate=run_args.simrate,
    dynamics_randomization=run_args.dyn_random,
    mirror=run_args.mirror,
    learn_gains=run_args.learn_gains,
    reward=run_args.reward,
    history=run_args.history,
    no_delta=run_args.no_delta,
    traj=run_args.traj,
    ik_baseline=run_args.ik_baseline
)
cassie_env = env_fn()
policy = torch.load(os.path.join(args.path, "actor.pt"))
if args.eval:
    policy.eval()
if hasattr(policy, 'init_hidden_state'):
    policy.init_hidden_state()

# TODO: make returning/save data in file inside function consist for all testing functions
def test_commands(cassie_env, policy, args):
    print("Testing speed and orient commands")
    if args.n_procs == 1:
        save_data = eval_commands(cassie_env, policy, num_steps=args.n_steps, num_commands=args.n_commands,
                max_speed=args.max_speed, min_speed=args.min_speed, num_iters=args.n_iter)
        np.save(os.path.join(args.path, "eval_commands.npy"), save_data)
    else:
        eval_commands_multi(env_fn, policy, num_steps=args.n_steps, num_commands=args.n_commands, max_speed=args.max_speed,
                min_speed=args.min_speed, num_iters=args.n_iter, num_procs=args.n_procs, filename=os.path.join(args.path, "eval_commands.npy"))

def test_perturbs(cassie_env, policy, args):
    print("Testing perturbations")
    if args.n_procs == 1:
        save_data = compute_perturbs(cassie_env, policy, wait_time=args.wait_time, perturb_duration=args.pert_dur, perturb_size=args.pert_size,
                    perturb_incr=args.pert_incr, perturb_body=args.pert_body, num_angles=args.num_angles)
    else:
        save_data = compute_perturbs_multi(env_fn, policy, wait_time=args.wait_time, perturb_duration=args.pert_dur, perturb_size=args.pert_size,
                    perturb_incr=args.pert_incr, perturb_body=args.pert_body, num_angles=args.num_angles, num_procs=args.n_procs)
    np.save(os.path.join(args.path, "eval_perturbs.npy"), save_data)

# If not command line arg, assume run all tests
if args.test == "full":
    print("Running full test")
    test_commands(cassie_env, policy, args)
    test_perturbs(cassie_env, policy, args)
elif args.test == "commands":
    test_commands(cassie_env, policy, args)
elif args.test == "perturb":
    test_perturbs(cassie_env, policy, args)
elif args.test == "mission":
    missions = ["straight", "curvy", "90_left", "90_right"]
    if not args.viz:
        print("Testing missions")
        save_data = []

        for mission in missions:
            print(mission + " mission:")
            cassie_env = CassiePlayground(traj=run_args.traj, clock_based=run_args.clock_based, state_est=run_args.state_est, dynamics_randomization=run_args.dyn_random, mission=mission)
            save_data.append(eval_mission(cassie_env, policy))
        np.save(os.path.join(args.path, "eval_missions.npy"), save_data)
    else:
        save_data = np.load(os.path.join(args.path, "eval_missions.npy"), allow_pickle=True)
        plot_mission_data(save_data, missions)
elif args.test == "sensitivity":
    print("Testing sensitivity")
    save_data = eval_sensitivity(cassie_env, policy, incr=args.sens_incr, hi_factor=args.hi_factor, lo_factor=args.lo_factor)
    print(save_data)
    np.save(os.path.join(args.path, "eval_sensitivity.npy"), save_data)
elif args.test == "compare":
    print("running compare")
    compare_pols(args.path, args.path2)

# vis_commands(cassie_env, policy, num_steps=200, num_commands=6, max_speed=3, min_speed=0)
# save_data = eval_commands(cassie_env, policy, num_steps=200, num_commands=2, max_speed=3, min_speed=0, num_iters=1)
# np.save("./test_eval_commands.npy", save_data)
# eval_commands_multi(env_fn, policy, num_steps=200, num_commands=4, max_speed=3, min_speed=0, num_iters=4, num_procs=4)

# report_stats("./test_eval_commands.npy")
