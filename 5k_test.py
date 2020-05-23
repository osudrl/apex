from rl.policies.actor import GaussianMLP_Actor
from tools.test_commands import *
from tools.eval_perturb import *
from tools.eval_mission import *
from tools.compare_pols import *
from tools.eval_sensitivity import *
from collections import OrderedDict
from util import env_factory
from cassie.cassiemujoco import CassieSim

import torch
import pickle
import os, sys, argparse
import numpy as np

# Visualizes a 5k test using the inputted env and policy for the given mission, terrain (xml model file)
# ground friction (3-long array), and foot mass (float)
def vis_5k_test(cassie_env, policy, mission, terrain, friction, foot_mass):
    # Reload CassieSim object for new terrain
    cassie_env.sim = CassieSim(terrain, reinit=True)
    # Load in mission
    with open(mission, 'rb') as mission_file:
        mission_commands = pickle.load(mission_file)
    mission_len = len(mission_commands['speed'])
    speeds = mission_commands['speed']
    orients = mission_commands['orient']
    state = cassie_env.reset_for_test()
    render_state = cassie_env.render()
    command_ind = 0
    while render_state and command_ind < mission_len:
        start = time.time()
        if (not cassie_env.vis.ispaused()):
            cassie_env.speed = speeds[command_ind]
            cassie_env.orient_add = orients[command_ind]
            action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
            state, reward, done, _ = cassie_env.step(action)
            command_ind += 1
        render_state = cassie_env.render()
        end = time.time()
        delaytime = max(0, 1000 / 30000 - (end-start))
        time.sleep(delaytime)

# Runs a 5k test using the inputted env and policy for the given mission, terrain (xml model file)
# ground friction (3-long array), and foot mass (float)
def sim_5k_test(cassie_env, policy, mission, terrain, friction, foot_mass):
    # Reload CassieSim object for new terrain
    cassie_env.sim = CassieSim(terrain, reinit=True)
    # Load in mission
    with open(mission, 'rb') as mission_file:
        mission_commands = pickle.load(mission_file)
    mission_len = len(mission_commands['speed'])
    speeds = mission_commands['speed']
    orients = mission_commands['orient']
    state = cassie_env.reset_for_test()
    for i in range(mission_len):
        cassie_env.speed = speeds[i]
        cassie_env.orient_add = orients[i]
        action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
        state, reward, done, _ = cassie_env.step(action)
        if cassie_env.sim.qpos()[2] < 0.4:  # Failed, reset and record force
            return False
    return True



# Get policy to test from args, load policy and env
parser = argparse.ArgumentParser()
# General args
parser.add_argument("--path", type=str, default="./trained_models/nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2", help="path to folder containing policy and run details")
parser.add_argument("--n_procs", type=int, default=4, help="Number of procs to use for multi-processing")
parser.add_argument("--lite", dest='full', default=False, action="store_true", help="run the lite test instead of full test")
parser.add_argument("--eval", default=True, action="store_false", help="Whether to call policy.eval() or not")
parser.add_argument("--vis", default=False, action="store_true", help="Whether to visualize test or not")

args = parser.parse_args()
run_args = pickle.load(open(os.path.join(args.path, "experiment.pkl"), "rb"))
# Make mirror False so that env_factory returns a regular wrap env function and not a symmetric env function that can be called to return
# a cassie environment (symmetric env cannot be called to make another env)
env_fn = env_factory(run_args.env_name, traj=run_args.traj, simrate=run_args.simrate, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, 
                    mirror=False, clock_based=run_args.clock_based, reward=run_args.reward, history=run_args.history)
cassie_env = env_fn()
policy = torch.load(os.path.join(args.path, "actor.pt"))
if args.eval:
    policy.eval()

model_dir = "./cassie/cassiemujoco"
mission_dir = "./cassie/missions/"
default_fric = np.array([1, 5e-3, 1e-4])
default_mass = .1498
if args.full:
    # Run all terrains and missions
    terrains = ["cassie.xml", "cassie_noise_terrain.xml"]
    missions = ["curvy", "straight", "90_left", "90_right"]
    mission_speeds = [0.5, 0.9, 1.4, 1.9, 2.3, 2.8]
    frictions = np.linspace(.8*default_fric, default_fric, 10)
    frictions = np.append(frictions, np.linspace(default_fric, 1.2*default_fric, 10)[1:])
    masses = np.linspace(.8*default_mass, default_mass, 10)
    masses = np.append(masses, np.linspace(default_mass, default_mass*1.2, 10)[1:])
else:
    # Only run flat, noisy, and hill terrain with straight and curvy missions
    terrains = ["cassie.xml", "cassie_noise_terrain.xml"]
    missions = ["curvy", "straight"]
    mission_speeds = [0.5, 0.9, 1.4, 1.9, 2.8]
    frictions = []
    masses = []

num_args = len(terrains)*len(missions)*len(frictions)*len(masses)
pass_data = [0]*num_args
terrain_data = [0]*num_args
mission_data = [0]*num_args
friction_data = [0]*num_args
mass_data = [0]*num_args
arg_count = 0
for terrain in terrains:
    for mission in missions:
        for mission_speed in mission_speeds:
            for friction in frictions:
                for mass in masses:
                    if args.vis:
                        vis_5k_test(cassie_env, policy, os.path.join(mission_dir, mission+"/command_trajectory_{}.pkl".format(mission_speed)), 
                                        os.path.join(model_dir, terrain), friction, mass)
                    else:
                        success = sim_5k_test(cassie_env, policy, os.path.join(mission_dir, mission+"/command_trajectory_{}.pkl".format(mission_speed)), 
                                        os.path.join(model_dir, terrain), friction, mass)
                        pass_data[arg_count] = success
                        terrain_data[arg_count] = terrain
                        mission_data[arg_count] = mission
                        friction_data[arg_count] = friction
                        mass_data[arg_count] = mass

if not args.vis:
    with open(os.path.join(args.path, "5k_test.pkl"), 'wb') as savefile:
        pickle.dump(pass_data, savefile)
        pickle.dump(terrain_data, savefile)
        pickle.dump(mission_data, savefile)
        pickle.dump(friction_data, savefile)
        pickle.dump(mass_data, savefile)