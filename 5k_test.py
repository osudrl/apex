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
import copy, time, psutil
import ray
import fpdf


@ray.remote
class test_worker(object):
    def __init__ (self, id_num, env_fn, policy, mission_data):
        self.id_num = id_num
        self.cassie_env = env_fn()
        self.policy = copy.deepcopy(policy)
        self.mission_data = mission_data    # Dictionary containing all mission data to be tested across all workers
        torch.set_num_threads(1)

    def test_5k(self, mission, mission_speed, terrain, friction, foot_mass):
        if ".npy" in terrain:
            self.cassie_env.sim = CassieSim("./cassie/cassiemujoco/cassie_hfield.xml", reinit=True)
            hfield_data = np.load(os.path.join("./cassie/cassiemujoco/terrains/", terrain))
            self.cassie_env.sim.set_hfield_data(hfield_data.flatten())
        else:
            self.cassie_env.sim = CassieSim("./cassie/cassiemujoco/cassie.xml", reinit=True)
            if not (".xml" in terrain):     # If not xml file, assume specify direction and angle for tilt
                direct, angle = terrain.split("_")
                if direct == "left":
                    floor_quat = euler2quat(z=0, x=np.deg2rad(angle), y=0)
                elif direct == "right":
                    floor_quat = euler2quat(z=0, x=np.deg2rad(-angle), y=0)
                elif direct == "up":
                    floor_quat = euler2quat(z=0, x=0, y=np.deg2rad(-angle))
                elif direct == "right":
                    floor_quat = euler2quat(z=0, x=0, y=np.deg2rad(angle))
                else:
                    print("Error: Terrain type not understood")
                    return 1
                self.cassie_env.sim.set_geom_quat(floor_quat, name="floor")
        
        self.cassie_env.sim.set_geom_friction(friction, "floor")
        self.cassie_env.sim.set_body_mass(foot_mass, "right-foot")
        self.cassie_env.sim.set_body_mass(foot_mass, "left-foot")
        # Load in mission
        # mission_path = os.path.join(mission, "command_trajectory_{}.pkl".format(mission_speed))
        # print("mission", mission)
        # print(mission_path)
        # with open(os.path.join("./cassie/missions/"+mission, "command_trajectory_{}.pkl".format(mission_speed)), 'rb') as mission_file:
            # mission_commands = pickle.load(mission_file)
        mission_commands = self.mission_data[mission+str(mission_speed)]
        mission_len = len(mission_commands['speed'])
        speeds = mission_commands['speed']
        orients = mission_commands['orient']
        state = self.cassie_env.reset_for_test()
        for i in range(mission_len):
            self.cassie_env.update_speed(speeds[i])
            self.cassie_env.orient_add = orients[i]
            with torch.no_grad():
                action = self.policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
            state = self.cassie_env.step_basic(action)
            if self.cassie_env.sim.qpos()[2] < 0.4:  # Failed, done testing
                # print("eval time: ", time.time()-start_t)
                return self.id_num, False, mission, mission_speed, terrain, friction, foot_mass
        # print("eval time: ", time.time()-start_t)
        return self.id_num, True, mission, mission_speed, terrain, friction, foot_mass

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
def sim_5k_test(cassie_env, policy, mission, mission_speed, terrain, friction, foot_mass):
    start_t = time.time()
    # Reload CassieSim object for new terrain
    cassie_env.sim = CassieSim(terrain, reinit=True)
    # Load in mission
    # with open(mission, 'rb') as mission_file:
        # mission_commands = pickle.load(mission_file)
    mission_commands = mission_dict[mission+str(mission_speed)]
    mission_len = len(mission_commands['speed'])
    print(mission_len)
    speeds = mission_commands['speed']
    orients = mission_commands['orient']
    state = cassie_env.reset_for_test()
    for i in range(mission_len):
        cassie_env.speed = speeds[i]
        cassie_env.orient_add = orients[i]
        with torch.no_grad():
            action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
        state = cassie_env.step_basic(action)
        if cassie_env.sim.qpos()[2] < 0.4:  # Failed, reset and record force
            print("eval time: ", time.time()-start_t)
            return False
    print("eval time: ", time.time()-start_t)
    return True

def calc_stats(pass_data, terrain_data, mission_data, mission_speed_data, friction_data, mass_data):
    test_len = len(pass_data)
    pass_data = np.array(pass_data)
    friction_data = np.array(friction_data)
    avg_pass = np.sum(pass_data)/test_len

    # Terrain breakdown
    terrain_names = set(terrain_data)
    terrain_dict = {}
    for terrain in terrain_names:
        terr_inds = [i for i, x in enumerate(terrain_data) if x == terrain]
        rel_pass = np.sum(pass_data[terr_inds]) / len(terr_inds)
        terrain_dict[os.path.basename(terrain)] = rel_pass

    # Mission breakdown
    # Compose mission with each speed, i.e. treat mission with a single speed as a single separate mission
    # NOTE: Assumes that EVERY mission is tested at EVERY speed. This is method is also probably pretty
    # inefficient, but fine for now
    mission_names = set(mission_data)
    speeds = set(mission_speed_data)
    # Compute ind list for every speed
    speed_inds = {}
    for speed in speeds:
        curr_inds = [i for i, x in enumerate(mission_speed_data) if x == speed]
        speed_inds[speed] = curr_inds
    mission_dict = {}
    for mission in mission_names:
        mission_inds = [i for i, x in enumerate(mission_data) if x == mission]
        miss_ind_set = set(mission_inds)
        for speed in speeds:
            speed_ind_set = set(speed_inds[speed])
            inter_inds = miss_ind_set.intersection(speed_ind_set)
            rel_pass = np.sum(pass_data[list(inter_inds)]) / len(inter_inds)
            mission_name = "{} {}".format(mission, speed)
            mission_dict[mission_name] = rel_pass
    
    # Friction breakdown
    frictions = np.unique(friction_data, axis=0)
    fric_dict = {}
    for fric in frictions:
        fric_inds = [i for i, x in enumerate(friction_data) if np.all(x == fric)]
        rel_pass = np.sum(pass_data[fric_inds]) / len(fric_inds)
        fric_dict[np.array2string(fric)] = rel_pass

    # Terrain breakdown
    masses = set(mass_data)
    mass_dict = {}
    for mass in masses:
        mass_inds = [i for i, x in enumerate(mass_data) if x == mass]
        rel_pass = np.sum(pass_data[mass_inds]) / len(mass_inds)
        mass_dict[str(round(mass, 6))] = rel_pass

    return avg_pass, terrain_dict, mission_dict, fric_dict, mass_dict

def report_stats(path):
    filepath = os.path.join(path, "5k_test.pkl")
    with open(filepath, "rb") as datafile:
        # pass_data, terrain_data, mission_data, friction_data, mass_data = pickle.load(datafile)
        data = pickle.load(datafile)
    
    # print(data)
    avg_pass, terrain_dict, mission_dict, fric_dict, mass_dict = calc_stats(*data)

     # Initial PDF setup
    pdf = fpdf.FPDF(format='letter', unit='in')
    pdf.add_page()
    pdf.set_font('Times', '', 12.0)
    
    # Effective page width, or just epw
    epw = pdf.w - 2*pdf.l_margin
    th = pdf.font_size
    # Set title
    pdf.set_font('Times', '', 18.0) 
    polname = os.path.basename(path)
    pdf.cell(epw, 2*th, "5K Test Report".format(polname), 0, 1, "C")
    pdf.ln(2*th)

    pdf.set_font('Times', '', 12.0)
    pdf.cell(epw, 2*th, "Policy: {}".format(polname), 0, 1)
    pdf.ln(2*th)

    pdf.cell(epw, 2*th, "Total Pass Rate: {}".format(avg_pass), 0, 1)
    pdf.ln(2*th)

    # Terrain breakdown
    pdf.cell(epw, 2*th, "Terrain Breakdown", 0, 1)
    pdf.ln(th)
    print_table(pdf, terrain_dict, "Terrain")
    pdf.ln(2*th)

    # Mission breakdown
    pdf.cell(epw, 2*th, "Mission Breakdown", 0, 1)
    pdf.ln(th)
    print_table(pdf, mission_dict, "Mission")
    pdf.ln(2*th)

    # Friction breakdown
    pdf.cell(epw, 2*th, "Friction Breakdown", 0, 1)
    pdf.ln(th)
    print_table(pdf, fric_dict, "Friction")
    pdf.ln(2*th)

    # Mission breakdown
    pdf.cell(epw, 2*th, "Foot Mass Breakdown", 0, 1)
    pdf.ln(th)
    print_table(pdf, mass_dict, "Foot Mass")
    pdf.ln(2*th)

    pdf.output(os.path.join(path, "5k_test.pdf"))

# Print table for the inputted data dictionary. Gives the neccessary width for the strings in the
# dict's keys, and gives rest of width the to values (rel pass rates)
def print_table(pdf, data_dict, title):
    epw = pdf.w - 2*pdf.l_margin
    th = pdf.font_size
    # print(data_dict.keys())
    # print(max(data_dict.keys(), key=len))
    name_width = map(pdf.get_string_width, data_dict.keys())
    col1_width = max(name_width) + .2
    col2_width = epw - col1_width
    start_x = pdf.get_x()
    start_y = pdf.get_y()

    pdf.cell(col1_width, 2*th, title, border=1, align="C")
    pdf.cell(col2_width, 2*th, "Relative Pass Rate", border=1, align="C")
    pdf.ln(2*th)

    for key in data_dict.keys():
        pdf.cell(col1_width, 2*th, key, border=1, align="C")
        pdf.cell(col2_width, 2*th, str(data_dict[key]), border=1, align="C")
        pdf.ln(2*th)



# Get policy to test from args, load policy and env
parser = argparse.ArgumentParser()
# General args
parser.add_argument("--path", type=str, default="./trained_models/nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2", help="path to folder containing policy and run details")
parser.add_argument("--n_procs", type=int, default=4, help="Number of procs to use for multi-processing")
parser.add_argument("--lite", dest='full', default=True, action="store_false", help="run the lite test instead of full test")
parser.add_argument("--eval", default=True, action="store_false", help="Whether to call policy.eval() or not")
parser.add_argument("--vis", default=False, action="store_true", help="Whether to visualize test or not")
parser.add_argument("--report", default=False, action="store_true", help="Whether to report stats or not")

args = parser.parse_args()
run_args = pickle.load(open(os.path.join(args.path, "experiment.pkl"), "rb"))

# Make mirror False so that env_factory returns a regular wrap env function and not a symmetric env function that can be called to return
# a cassie environment (symmetric env cannot be called to make another env)
if hasattr(run_args, 'simrate'):
    env_fn = env_factory(run_args.env_name, traj=run_args.traj, simrate=run_args.simrate, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, 
                    mirror=False, clock_based=run_args.clock_based, reward=run_args.reward, history=run_args.history)
else:
    env_fn = env_factory(run_args.env_name, traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, 
                    mirror=False, clock_based=run_args.clock_based, reward=run_args.reward, history=run_args.history)
cassie_env = env_fn()
policy = torch.load(os.path.join(args.path, "actor.pt"))
if args.eval:
    policy.eval()
if hasattr(policy, 'init_hidden_state'):
    policy.init_hidden_state()
num_procs = args.n_procs
print("num cpus:", psutil.cpu_count())
torch.set_num_threads(1)

model_dir = "./cassie/cassiemujoco"
mission_dir = "./cassie/missions/"
default_fric = np.array([1, 5e-3, 1e-4])
default_mass = .1498
if args.full:
    print("Running full test")
    # Run all terrains and missions
    terrains = ["cassie.xml", "noise1.npy", "noise2.npy", "noise3.npy", "rand_hill1.npy", "rand_hill2.npy", "rand_hill3.npy",
                "left_3", "right_3", "up_3", "down_3"]
    missions = ["curvy", "straight", "90_left", "90_right"]
    mission_speeds = [0.5, 0.9, 1.4, 1.9, 2.3, 2.8]
    frictions = np.linspace(.8*default_fric, default_fric, 10)
    frictions = np.concatenate((frictions, np.linspace(default_fric, 1.2*default_fric, 10)[1:]), axis=0)
    masses = np.linspace(.8*default_mass, default_mass, 10)
    masses = np.append(masses, np.linspace(default_mass, default_mass*1.2, 10)[1:])
else:
    print("Running lite test")
    # Only run flat, noisy, and hill terrain with straight and curvy missions
    terrains = ["cassie.xml", "noise1.npy", "rand_hill1.npy"]
    missions = ["curvy", "straight"]
    mission_speeds = [0.5, 0.9, 1.4, 1.9, 2.8]
    frictions = [default_fric]
    masses = [default_mass]

# Load missions
mission_dict = {}
for mission in missions:
    for speed in mission_speeds:
        with open(os.path.join(mission_dir, mission+"/command_trajectory_{}.pkl".format(speed)), 'rb') as mission_file:
            mission_dict[mission+str(speed)] = pickle.load(mission_file)         

# Make list of test args
test_args = [(mission, mission_speed, terrain, friction, mass) \
            for terrain in terrains for mission in missions for mission_speed in mission_speeds for friction in frictions for mass in masses]
# test_args = test_args[0:4] # For debugging. Makes n_procs > 4 fail obbiously


# If visualizing, only use 1 process, don't start any workers
if args.vis:
    for arg in test_args:
        print("Testing ", arg)
        vis_5k_test(cassie_env, policy, *arg)
else:
    
    # Make and start all workers
    print("Using {} processes".format(num_procs))
    ray.shutdown()
    ray.init(num_cpus=num_procs)
    workers = [test_worker.remote(i, env_fn, policy, mission_dict) for i in range(num_procs)]
    print("made workers")
    result_ids = [workers[i].test_5k.remote(*test_args[i]) for i in range(num_procs)]
    print("started workers")
    curr_arg_ind = num_procs

    # num_args = len(terrains)*len(missions)*len(mission_speeds)*len(frictions)*len(masses)
    num_args = len(test_args) 
    pass_data = [0]*num_args
    terrain_data = [0]*num_args
    mission_data = [0]*num_args
    mission_speed_data = [0]*num_args
    friction_data = [0]*num_args
    mass_data = [0]*num_args
    arg_count = 0
    sys.stdout.write("Finished {} out of {} tests".format(arg_count, num_args))
    sys.stdout.flush()
    start_t = time.time()
    while result_ids:
        done_id = ray.wait(result_ids, num_returns=1, timeout=None)[0][0]
        worker_id, success, mission, mission_speed, terrain, friction, mass = ray.get(done_id)
        pass_data[arg_count] = success
        terrain_data[arg_count] = terrain
        mission_data[arg_count] = mission
        mission_speed_data[arg_count] = mission_speed
        friction_data[arg_count] = friction
        mass_data[arg_count] = mass
        result_ids.remove(done_id)
        if curr_arg_ind < num_args:
            result_ids.append(workers[worker_id].test_5k.remote(*test_args[curr_arg_ind]))
        curr_arg_ind += 1
        arg_count += 1
        elapsed_time = time.time() - start_t
        time_left = elapsed_time/arg_count * (num_args-arg_count)
        sys.stdout.write("\rFinished {} out of {} tests. {:.1f}s elapsed, {:.1f}s left".format(arg_count, num_args, elapsed_time, time_left))
        sys.stdout.flush()
        # TODO: Add progress bar and estimated time left
    print()
    print("Total time: ", time.time() - start_t)


if not args.vis:
    ray.shutdown()
    with open(os.path.join(args.path, "5k_test.pkl"), 'wb') as savefile:
        pickle.dump([pass_data, terrain_data, mission_data, mission_speed_data, friction_data, mass_data], savefile)

    report_stats(args.path)


