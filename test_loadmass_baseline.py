from rl.policies.actor import Gaussian_FF_Actor
from tools.test_commands import *
from tools.eval_perturb import *
from util import env_factory

import torch
import pickle
import os, sys, argparse
import numpy as np
from cassie import CassieEnv_accel_nofy


# Get policy to test from args, load policy and env
parser = argparse.ArgumentParser()
# General args
parser.add_argument("--path", type=str, default="./trained_models/nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2", help="path to folder containing policy and run details")
parser.add_argument("--n_procs", type=int, default=56, help="Number of procs to use for multi-processing")
parser.add_argument("--eval", default=True, action="store_false", help="Whether to call policy.eval() or not")
# Test Commands args
parser.add_argument("--n_steps", type=int, default=300, help="Number of steps to for a full command cycle (1 speed change and 1 orientation change)")
parser.add_argument("--n_commands", type=int, dxefault=1, help="Number of commands in a single test iteration")
parser.add_argument("--max_speed", type=float, default=4.0, help="Maximum allowable speed to test")
parser.add_argument("--min_speed", type=float, default=0.0, help="Minimum allowable speed to test")
parser.add_argument("--n_iter", type=int, default=5000, help="Number of command cycles to test")
# Test Perturbs args
parser.add_argument("--wait_time", type=float, default=3.0, help="How long to wait after perturb to count as success")
parser.add_argument("--pert_dur", type=float, default=0.2, help="How long to apply perturbation")
parser.add_argument("--pert_size", type=float, default=50, help="Size of perturbation to start sweep from")
parser.add_argument("--pert_incr", type=float, default=10.0, help="How much to increment the perturbation size after each success")
parser.add_argument("--pert_body", type=str, default="cassie-pelvis", help="Body to apply perturbation to")
parser.add_argument("--num_angles", type=int, default=36, help="How many angles to test (angles are evenly divided into 2*pi)")

args = parser.parse_args()
run_args = pickle.load(open(os.path.join(args.path, "experiment.pkl"), "rb"))

# env_fn = env_factory(run_args.env_name, traj=run_args.traj, simrate=run_args.simrate, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, 
#                     mirror=True, clock_based=run_args.clock_based, reward=run_args.reward, history=run_args.history)

policy = torch.load(os.path.join(args.path, "actor.pt"))
if args.eval:
    policy.eval()
if hasattr(policy, 'init_hidden_state'):
    policy.init_hidden_state()

model_list = ["cassie.xml", "cassie_tray_box.xml", "cassie_cart_soft.xml", "cassie_carry_pole.xml", "cassie_jug_spring.xml"]
# model_list = ["cassie_jug_spring.xml"]

for model in model_list:
    env_fn = partial(CassieEnv_accel_nofy, simrate=run_args.simrate, dynamics_randomization=False, reward=run_args.reward, history=run_args.history, model=model, reinit=True)

    eval_commands_multi(env_fn, policy, num_steps=args.n_steps, num_commands=args.n_commands, max_speed=args.max_speed, 
                    min_speed=args.min_speed, num_iters=args.n_iter, num_procs=args.n_procs, filename=os.path.join(args.path, 
                    "eval_commands_{}.npy".format(model[:-4])))

    save_data = compute_perturbs_multi(env_fn, policy, wait_time=args.wait_time, perturb_duration=args.pert_dur, perturb_size=args.pert_size, 
                        perturb_incr=args.pert_incr, perturb_body=args.pert_body, num_angles=args.num_angles, num_procs=args.n_procs)
    np.save(os.path.join(args.path, "eval_perturbs_{}.npy".format(model[:-4])), save_data)