import torch
import sys, pickle, argparse
from util import create_logger, env_factory
from cassie.quaternion_function import *
import numpy as np
import matplotlib.pyplot as plt
from cassie import CassieEnv, CassieMinEnv, CassiePlayground, CassieStandingEnv, CassieEnv_noaccel_footdist_omniscient, CassieEnv_footdist, CassieEnv_noaccel_footdist, CassieEnv_noaccel_footdist_nojoint, CassieEnv_novel_footdist, CassieEnv_mininput, CassieEnv_mininput_vel_sidestep

@torch.no_grad()
def run_sample_pol(env, policy, num_steps):
    qpos = np.zeros((num_steps*env.simrate, 35))
    pol_input = np.zeros((num_steps*env.simrate, env._obs))
    env.sim.full_reset()
    env.speed = 0
    env.phase_add = 1

    for i in range(num_steps):
        state = env.get_full_state()
        action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
        for j in range(env.simrate):
            qpos[j+env.simrate*i, :] = env.sim.qpos()
            pol_input[j+env.simrate*i, :] = env.get_full_state()
            env.step_sim_basic(action)

    return qpos, pol_input

def run_sample_zero(env, policy, num_steps):
    qpos = np.zeros((num_steps*env.simrate, 35))
    pol_input = np.zeros((num_steps*env.simrate, env._obs))
    env.sim.full_reset()
    env.P = np.zeros(5)
    env.D = np.zeros(5)
    env.speed = 0
    env.phase_add = 1
    zero_ctrl = np.zeros(10)

    for i in range(num_steps):
        for j in range(env.simrate):
            qpos[j+env.simrate*i, :] = env.sim.qpos()
            pol_input[j+env.simrate*i, :] = env.get_full_state()
            env.step_sim_basic(zero_ctrl)

    return qpos, pol_input

def run_sample_constant(env, policy, num_steps):
    qpos = np.zeros((num_steps*env.simrate, 35))
    pol_input = np.zeros((num_steps*env.simrate, env._obs))
    env.sim.full_reset()
    env.speed = 0
    env.phase_add = 1
    constant_ctrl = np.array([0.0045, 0, 0.4973, -1.1997, -1.5968, -0.0045, 0, 0.4973, -1.1997, -1.5968])

    for i in range(num_steps):
        for j in range(env.simrate):
            qpos[j+env.simrate*i, :] = env.sim.qpos()
            pol_input[j+env.simrate*i, :] = env.get_full_state()
            env.step_sim_basic(constant_ctrl)

    return qpos, pol_input

def collect_data(args, sample_fn):
    run_args = pickle.load(open(args.path + "experiment.pkl", "rb"))

    policy = torch.load(args.path + "actor.pt")
    policy.eval()

    print("env name: ", run_args.env_name)
    env_name = run_args.env_name

    run_args.dyn_random = True
    args.reward = run_args.reward
    env_fn = env_factory(env_name, traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, phase_based=run_args.phase_based, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
    # if env_name == "Cassie-v0":
    #     env = CassieEnv(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, phase_based=run_args.phase_based, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
    # elif env_name == "CassieMin-v0":
    #     env = CassieMinEnv(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, phase_based=run_args.phase_based, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
    # elif env_name == "CassiePlayground-v0":
    #     env = CassiePlayground(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history, mission=args.mission)
    # elif env_name == "CassieNoaccelFootDistOmniscient":
    #     env = CassieEnv_noaccel_footdist_omniscient(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
    # elif env_name == "CassieFootDist":
    #     env = CassieEnv_footdist(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
    # elif env_name == "CassieNoaccelFootDist":
    #     env = CassieEnv_noaccel_footdist(traj=run_args.traj, simrate=run_args.simrate, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
    # elif env_name == "CassieNoaccelFootDistNojoint":
    #     env = CassieEnv_noaccel_footdist_nojoint(traj=run_args.traj, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, reward=args.reward, history=run_args.history)
    # elif env_name == "CassieNovelFootDist":
    #     env = CassieEnv_novel_footdist(traj=run_args.traj, simrate=run_args.simrate, clock_based=run_args.clock_based, state_est=run_args.state_est, dynamics_randomization=run_args.dyn_random, no_delta=run_args.no_delta, reward=args.reward, history=run_args.history)
    # elif env_name == "CassieMinInput":
    #     env = CassieEnv_mininput(traj=run_args.traj, simrate=run_args.simrate, clock_based=run_args.clock_based, state_est=run_args.state_est, dynamics_randomization=run_args.dyn_random, no_delta=run_args.no_delta, learn_gains=run_args.learn_gains, reward=args.reward, history=run_args.history)
    # elif env_name == "CassieMinInputVelSidestep":
    #     env = CassieEnv_mininput_vel_sidestep(traj=run_args.traj, simrate=run_args.simrate, clock_based=run_args.clock_based, state_est=run_args.state_est, dynamics_randomization=run_args.dyn_random, no_delta=run_args.no_delta, reward=args.reward, history=run_args.history)
    # else:
    #     env = CassieStandingEnv(state_est=run_args.state_est)

    # Normal data
    print("Doing normal data")
    env = env_fn()
    normal_data = sample_fn(env, policy, args.num_steps)

    # Control test
    print("Doing control data")
    env = env_fn()
    control_data = sample_fn(env, policy, args.num_steps)
    # print(control_data[0].shape, control_data[1].shape)
    # print(normal_data[0]-control_data[0])

    # Damping Test
    print("Doing damping data")
    env = env_fn()
    low_damp = [a for a, b in env.damp_range]
    env.sim.set_dof_damping(low_damp)
    env.sim.set_const()
    low_damp_data = sample_fn(env, policy, args.num_steps)
    env = env_fn()
    high_damp = [b for a, b in env.damp_range]
    env.sim.set_dof_damping(high_damp)
    env.sim.set_const()
    high_damp_data = sample_fn(env, policy, args.num_steps)
    env.sim.set_dof_damping(env.default_damping)

    # Mass Test
    print("Doing mass data")
    env = env_fn()
    low_mass = [a for a, b in env.mass_range]
    env.sim.set_body_mass(low_mass)
    env.sim.set_const()
    low_mass_data = sample_fn(env, policy, args.num_steps)
    env = env_fn()
    high_mass = [b for a, b in env.mass_range]
    env.sim.set_body_mass(high_mass)
    env.sim.set_const()
    high_mass_data = sample_fn(env, policy, args.num_steps)
    env.sim.set_body_mass(env.default_mass)

    # Friction Test
    print("Doing friction data")
    env = env_fn()
    low_fric = [0.6, 1e-4, 5e-5]
    env.sim.set_geom_friction(low_fric, "floor")
    env.sim.set_const()
    low_fric_data = sample_fn(env, policy, args.num_steps)
    env = env_fn()
    high_fric = [1.2, 1e-2, 5e-4]
    env.sim.set_geom_friction(high_fric, "floor")
    env.sim.set_const()
    high_fric_data = sample_fn(env, policy, args.num_steps)

    # All Test
    print("Doing all data")
    env = env_fn()
    env.sim.set_dof_damping(low_damp)
    env.sim.set_body_mass(low_mass)
    env.sim.set_geom_friction(low_fric, "floor")
    env.sim.set_const()
    low_all_data = sample_fn(env, policy, args.num_steps)
    env = env_fn()
    env.sim.set_dof_damping(high_damp)
    env.sim.set_body_mass(high_mass)
    env.sim.set_geom_friction(high_fric, "floor")
    env.sim.set_const()
    high_all_data = sample_fn(env, policy, args.num_steps)

    save_dict = {"normal":normal_data, "control":control_data, "l_damp":low_damp_data, "h_damp":high_damp_data, 
                                "l_mass":low_mass_data, "h_mass":high_mass_data, "l_fric":low_fric_data, "h_fric":high_fric_data,
                                "l_all":low_all_data, "h_all":high_all_data}
    pickle.dump(save_dict, open("./"+args.save_name+".pkl", "wb"))
    # np.savez("./state_diff.npz", normal=normal_data, control=control_data, l_damp=low_damp_data, h_damp=high_damp_data, 
    #                             l_mass=low_mass_data, h_mass=high_mass_data, l_fric=low_fric_data, h_fric=high_fric_data,
    #                             l_all=low_all_data, h_all=high_all_data)
    


parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, default="./trained_models/ppo/Cassie-v0/7b7e24-seed0/", help="path to folder containing policy and run details")
parser.add_argument("--num_steps", type=int, default=40, help="Number of policy update steps to run.")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--samp_fn", type=str, default="run_sample_pol", help="Which function to use when collecting data")
parser.add_argument("--save_name", type=str, default="state_diff", help="Name of the file to save data/plot to")


args = parser.parse_args()

if not args.plot:
    print("Collecting data using {} function".format(args.samp_fn))
    collect_data(args, globals()[args.samp_fn])
else:
    print("Plotting data")
    save_dict = pickle.load(open("./"+args.save_name+".pkl", "rb"))

    normal = save_dict["normal"]
    control = save_dict["control"]
    l_damp = save_dict["l_damp"]
    h_damp = save_dict["h_damp"]
    l_mass = save_dict["l_mass"]
    h_mass = save_dict["h_mass"]
    l_fric = save_dict["l_fric"]
    h_fric = save_dict["h_fric"]
    l_all = save_dict["l_all"]
    h_all = save_dict["h_all"]
    
    control_qpos_diff = np.linalg.norm(normal[0] - control[0], axis=1)
    control_input_diff = np.linalg.norm(normal[1] - control[1], axis=1)

    l_damp_qpos_diff = np.linalg.norm(normal[0] - l_damp[0], axis=1)
    l_damp_input_diff = np.linalg.norm(normal[1] - l_damp[1], axis=1)
    h_damp_qpos_diff = np.linalg.norm(normal[0] - h_damp[0], axis=1)
    h_damp_input_diff = np.linalg.norm(normal[1] - h_damp[1], axis=1)

    l_mass_qpos_diff = np.linalg.norm(normal[0] - l_mass[0], axis=1)
    l_mass_input_diff = np.linalg.norm(normal[1] - l_mass[1], axis=1)
    h_mass_qpos_diff = np.linalg.norm(normal[0] - h_mass[0], axis=1)
    h_mass_input_diff = np.linalg.norm(normal[1] - h_mass[1], axis=1)

    l_fric_qpos_diff = np.linalg.norm(normal[0] - l_fric[0], axis=1)
    l_fric_input_diff = np.linalg.norm(normal[1] - l_fric[1], axis=1)
    h_fric_qpos_diff = np.linalg.norm(normal[0] - h_fric[0], axis=1)
    h_fric_input_diff = np.linalg.norm(normal[1] - h_fric[1], axis=1)

    l_all_qpos_diff = np.linalg.norm(normal[0] - l_all[0], axis=1)
    l_all_input_diff = np.linalg.norm(normal[1] - l_all[1], axis=1)
    h_all_qpos_diff = np.linalg.norm(normal[0] - h_all[0], axis=1)
    h_all_input_diff = np.linalg.norm(normal[1] - h_all[1], axis=1)

    l_qpos_data = [l_damp_qpos_diff, l_mass_qpos_diff, l_fric_qpos_diff, l_all_qpos_diff]
    l_input_data = [l_damp_input_diff, l_mass_input_diff, l_fric_input_diff, l_all_input_diff]
    h_qpos_data = [h_damp_qpos_diff, h_mass_qpos_diff, h_fric_qpos_diff, h_all_qpos_diff]
    h_input_data = [h_damp_input_diff, h_mass_input_diff, h_fric_input_diff, h_all_input_diff]

    fig, ax = plt.subplots(2, 1, figsize=(10, 15))
    types = ["damp", "mass", "fric", "all"]
    direct = ["Low", "High"]
    t = np.linspace(0, len(control_qpos_diff)*0.0005, len(control_qpos_diff))
    ax[0].plot(t, control_qpos_diff, label="control")
    for i in range(4):
        ax[0].plot(t, l_qpos_data[i], label="Low "+types[i])
        ax[0].plot(t, h_qpos_data[i], label="High "+types[i])
    ax[0].set_ylabel("Norm State Diff")
    ax[0].set_title("Qpos Difference")
    ax[0].legend(loc="upper left")

    ax[1].plot(t, control_input_diff, label="control")
    for i in range(4):
        ax[1].plot(t, l_input_data[i], label="Low "+types[i])
        ax[1].plot(t, h_input_data[i], label="High "+types[i])
    ax[1].set_ylabel("Norm Input Diff")
    ax[1].set_title("Input Difference")
    ax[1].legend(loc="upper left")
    ax[1].set_xlabel("Time (sec)")

    # plt.show()
    plt.tight_layout()
    plt.savefig("./"+args.save_name+".png")
    



