import pickle
from matplotlib import pyplot as plt
import numpy as np
import cassie
import time
from tempfile import TemporaryFile
import os, re
import argparse
import torch
from util import *
from cassie.cassiemujoco.cassiemujoco_ctypes import *
from cassie.cassiemujoco import pd_in_t, state_out_t, CassieSim
from cassie.quaternion_function import *

from cassie.cassiemujocoQPOS import CassieQposEst

def plot_data(sync, save):
    data = np.load("mj_diffdata_sync{}.npz".format(sync), allow_pickle=True)

    pel_pos = data["pel_pos"]
    pel_orient = data["pel_orient"]
    pel_trans_vel = data["pel_trans_vel"]
    pel_rot_vel = data["pel_rot_vel"]
    pel_accel = data["pel_accel"]
    motors_pos = data["motors_pos"]
    motors_vel = data["motors_vel"]
    joints_pos = data["joints_pos"]
    joints_vel = data["joints_vel"]
    torques = data["torques"]
    l_foot_pos = data["l_foot_pos"]
    l_foot_orient = data["l_foot_orient"]
    l_foot_trans_vel = data["l_foot_trans_vel"]
    l_foot_rot_vel = data["l_foot_rot_vel"]
    l_foot_toe_force = data["l_foot_toe_force"]
    l_foot_heel_force = data["l_foot_heel_force"]
    r_foot_pos = data["r_foot_pos"]
    r_foot_orient = data["r_foot_orient"]
    r_foot_trans_vel = data["r_foot_trans_vel"]
    r_foot_rot_vel = data["r_foot_rot_vel"]
    r_foot_toe_force = data["r_foot_toe_force"]
    r_foot_heel_force = data["r_foot_heel_force"]

    total_diff = np.concatenate((pel_pos, pel_orient, pel_trans_vel, pel_rot_vel, pel_accel,
                    motors_pos, motors_vel, joints_pos, joints_vel, torques,
                    l_foot_pos, l_foot_orient, l_foot_trans_vel, l_foot_rot_vel, l_foot_toe_force, l_foot_heel_force,
                    r_foot_pos, r_foot_orient, r_foot_trans_vel, r_foot_rot_vel, r_foot_toe_force, r_foot_heel_force), axis=1)
    total_diff = np.linalg.norm(total_diff, axis=1)
    

    input_diff = np.concatenate((pel_orient, pel_trans_vel, pel_rot_vel, 
                    motors_pos, motors_vel, joints_pos[:, [0,1,3,4]], joints_vel[:, [0,1,3,4]], 
                    l_foot_pos, r_foot_pos), axis=1)
    input_diff = np.linalg.norm(input_diff, axis=1)

    print(np.median(total_diff))
    print(np.median(input_diff))
    print(np.max(pel_pos))

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    numStates = total_diff.shape[0]
    x_start = 100#10000
    x_end = -1#15000
    t = np.linspace(0, numStates*0.0005, numStates)
    ax[0].plot(t[x_start:x_end], total_diff[x_start:x_end])
    ax[0].set_ylabel("Diff Norm")
    ax[0].set_title("Total Difference")
    # ax[0].set_ylim(200, 350)
    ax[1].plot(t[x_start:x_end], input_diff[x_start:x_end])
    ax[1].set_xlabel("Time (sec)")
    ax[1].set_ylabel("Diff Norm")
    ax[1].set_title("Input Difference")
    # ax[1].set_ylim(0, 8)

    plt.tight_layout()
    if save:
        plt.savefig("./diffdata_total_sync{}.png".format(sync))
    else:
        plt.show()


    vis = {"pel_orient": True,
        "pel_trans_vel": True,
        "pel_rot_vel": False,
        "motors_pos": True,
        "motors_vel": True,
        "joints_pos": True,
        "joints_vel": True,
        "l_foot_pos": True,
        "r_foot_pos": True}

    vis = {"pel_pos" : True, 
           "pel_orient" : True, 
           "pel_trans_vel" : True,
           "pel_rot_vel" : False, 
           "pel_accel" : False,
           "motors_pos" : True,
           "motors_vel" : True,
           "joints_pos" : True,
           "joints_vel" : True,
           "torques" : False,
           "l_foot_pos" : True,
           "l_foot_orient" : False,
           "l_foot_trans_vel" : False,
           "l_foot_rot_vel" : False,
           "l_foot_toe_force" : False,
           "l_foot_heel_force" : False,
           "r_foot_pos" : True,
           "r_foot_orient" : False,
           "r_foot_trans_vel" : False,
           "r_foot_rot_vel" : False,
           "r_foot_toe_force" : False,
           "r_foot_heel_force" : False}

    plotnum = sum(vis.values())

    row = plotnum
    idx = 0
    fig, ax = plt.subplots(row, 1, figsize=(10, 2*row), sharex=True)

    # x_start = 10000
    # x_end = 20000

    for key in vis.keys():
        if vis[key]:
            print(key)
            curr_data = data[key]
            curr_data = np.linalg.norm(curr_data, axis=1)
            ax[idx].plot(t[x_start:x_end], curr_data[x_start:x_end])
            ax[idx].set_ylabel("Diff Norm")
            ax[idx].set_title(key)
            idx += 1
    
    ax[idx-1].set_xlabel("Time (sec)")
    plt.tight_layout()
    if save:
        plt.savefig("./stateest_filter_indiv_sync{}.png".format(sync))
    else:
        plt.show()

@torch.no_grad()
def process_data(policy, cassie_env, delay_env, sync, num_steps):

    # Reset envs and env setup
    state = cassie_env.reset_for_test()
    delay_env.reset_for_test()
    cassie_env.speed = 0
    cassie_env.phase_add = 1
    delay_env.speed = 0
    delay_env.phase_add = 1
    qpos_est = CassieQposEst()  # For optimizing closed loop
    mj_u = pd_in_t()

    # Take 50 steps in orig env first
    for i in range(50):
        action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
        state = cassie_env.step_basic(action)

    # mj_sim = CassieSim("./cassie/cassiemujoco/cassie.xml")
    
    # mj_sim.step_pd(mj_u)

    P = np.array([100,  100,  88,  96,  50]) 
    D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])
    # P[0] *= 0.8
    # D[0] *= 0.8
    # P[1] *= 0.8
    # D[1] *= 0.8
    for i in range(5):
        mj_u.leftLeg.motorPd.pGain[i]  = P[i]
        mj_u.rightLeg.motorPd.pGain[i] = P[i]

        mj_u.leftLeg.motorPd.dGain[i]  = D[i]
        mj_u.rightLeg.motorPd.dGain[i] = D[i]

    mj_count = 1
    simrate = cassie_env.simrate

    pel_pos                 = np.zeros((num_steps*simrate, 3))
    pel_orient              = np.zeros((num_steps*simrate, 4))
    pel_trans_vel           = np.zeros((num_steps*simrate, 3))
    pel_rot_vel             = np.zeros((num_steps*simrate, 3))
    pel_accel               = np.zeros((num_steps*simrate, 3))

    motors_pos              = np.zeros((num_steps*simrate, 10))
    motors_vel              = np.zeros((num_steps*simrate, 10))
    joints_pos              = np.zeros((num_steps*simrate, 6))
    joints_vel              = np.zeros((num_steps*simrate, 6))
    torques                 = np.zeros((num_steps*simrate, 10))

    l_foot_pos              = np.zeros((num_steps*simrate, 3))
    l_foot_orient           = np.zeros((num_steps*simrate, 4))
    l_foot_trans_vel        = np.zeros((num_steps*simrate, 3))
    l_foot_rot_vel          = np.zeros((num_steps*simrate, 3))
    l_foot_toe_force        = np.zeros((num_steps*simrate, 3))
    l_foot_heel_force       = np.zeros((num_steps*simrate, 3))

    r_foot_pos              = np.zeros((num_steps*simrate, 3))
    r_foot_orient           = np.zeros((num_steps*simrate, 4))
    r_foot_trans_vel        = np.zeros((num_steps*simrate, 3))
    r_foot_rot_vel          = np.zeros((num_steps*simrate, 3))
    r_foot_toe_force        = np.zeros((num_steps*simrate, 3))
    r_foot_heel_force       = np.zeros((num_steps*simrate, 3))

    for j in range(num_steps):
        state = cassie_env.get_full_state()
        action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
        sim_target = action + cassie_env.offset
        mj_u = pd_in_t()
        for i in range(5):
            mj_u.leftLeg.motorPd.pGain[i]  = P[i]
            mj_u.rightLeg.motorPd.pGain[i] = P[i]

            mj_u.leftLeg.motorPd.dGain[i]  = D[i]
            mj_u.rightLeg.motorPd.dGain[i] = D[i]

            mj_u.leftLeg.motorPd.torque[i]  = 0 # Feedforward torque
            mj_u.rightLeg.motorPd.torque[i] = 0 

            mj_u.leftLeg.motorPd.pTarget[i]  = sim_target[i]#action[i] + cassie_env.offset[i]
            mj_u.rightLeg.motorPd.pTarget[i] = sim_target[i+5]#action[i + 5] + cassie_env.offset[i + 5]

            mj_u.leftLeg.motorPd.dTarget[i]  = 0
            mj_u.rightLeg.motorPd.dTarget[i] = 0

        # print(cassie_env.sim.qpos()[2])
        for i in range(cassie_env.simrate):
            if mj_count == sync:
                # Set qpos and qvel
                s = cassie_env.cassie_state
                curr_pel_pos = s.pelvis.position[:]
                # curr_pel_pos[2] -= s.terrain.height
                curr_qpos =  qpos_est.getQpos(curr_pel_pos, s.pelvis.orientation[:], 
                                    s.motor.position[:], s.joint.position[:])
                zero3 = np.zeros(3)
                curr_qvel = np.concatenate((s.pelvis.translationalVelocity[:], s.pelvis.rotationalVelocity[:],
                        s.motor.velocity[0:3], zero3, [s.motor.velocity[3]], s.joint.velocity[0:2],
                        zero3, s.motor.velocity[4:8], zero3, [s.motor.velocity[8]], 
                        s.joint.velocity[3:5], zero3, [s.motor.velocity[9]]))
                # print(curr_qvel.shape)
                real_pos = np.array(cassie_env.sim.qpos())
                real_vel = np.array(cassie_env.sim.qvel())
                real_qacc = np.array(cassie_env.sim.qacc())
                # real_act_vel = np.array(cassie_env.sim.get_act_vel())
                # delay_env.sim.set_qpos(curr_qpos)
                delay_env.sim.set_qvel(curr_qvel)
                delay_env.sim.set_qpos(real_pos)
                # delay_env.sim.set_qvel(real_vel)
                # delay_env.sim.set_qacc(real_qacc)
                real_sensor = np.array(cassie_env.sim.get_sensordata())
                print("sensor: {}\t qpos: {}".format(real_sensor[0:3], real_pos[7:10]))
                # print(real_pos - curr_qpos)
                # print(real_vel - curr_qvel)
                delay_sensor = np.array(delay_env.sim.get_sensordata())
                # fake_sensor = np.concatenate((s.motor.position[0:5], s.joint.position[0:3], s.motor.position[5:], s.joint.position[3:],
                                # s.pelvis.orientation[:], s.pelvis.rotationalVelocity[:], s.pelvis.translationalAcceleration[:], real_sensor[-3:]))
                fake_sensor = np.concatenate((real_pos[[7,8,9,14,20]], s.joint.position[0:3], real_pos[[21,22,23,28,34]], s.joint.position[3:],
                                s.pelvis.orientation[:], s.pelvis.rotationalVelocity[:], s.pelvis.translationalAcceleration[:], real_sensor[-3:]))
                # print("real:", real_sensor)
                # print("delay:", delay_sensor)
                # print("fake:", fake_sensor)
                # print(real_pos[7:10])
                # exit()
                # real_out = cassie_env.get_cassie_out()

                # delay_env.sim.set_sensordata(real_sensor)
                # delay_env.sim.set_sensordata(fake_sensor)
                # delay_env.sim.copy_mjd(cassie_env.sim)
                # delay_env.sim.copy_state_est(cassie_env.sim)

                # delay_env.sim.copy(cassie_env.sim)
                # delay_env.sim.copy_just_sim(cassie_env.sim)
                # delay_env.sim.set_act_vel(real_act_vel)
                # print("set state")

                # Set filter and torque delay state
                joint_filter = cassie_env.sim.get_joint_filter()
                j_x = np.zeros(6*4)
                j_y = np.zeros(6*3)
                for k in range(6):
                    for l in range(4):
                        j_x[k*4 +l] = joint_filter[k].x[l]
                        # j_x[k*4 +l] = s.joint.position[k]
                        # print(joint_filter[k].x[l], s.joint.position[k])
                    for l in range(3):
                        j_y[k*3 +l] = joint_filter[k].y[l]
                        # j_y[k*3 +l] = s.joint.velocity[k]
                        # print(joint_filter[k].y[l], s.joint.velocity[k])
                # print("in python:", j_x)
                # delay_env.sim.set_joint_filter2(j_x, j_y)
                # delay_env.sim.set_joint_filter(joint_filter)
                drive_filter = cassie_env.sim.get_drive_filter()
                # d_x = np.zeros(10*9, dtype=np.int32)
                d_x = [0] * 90
                # print("in python")
                for k in range(10):
                    # print(drive_filter[k].x[:])
                    for l in range(9):
                        # print(type(drive_filter[k].x[l]))
                        d_x[k*9 +l] = drive_filter[k].x[l]
                        # d_x[k*9 +l] = drive_filter[k].x[l]
                # print("in python:", d_x)
                # print(type(d_x))
                # delay_env.sim.set_drive_filter2(d_x)
                # exit()
                # delay_env.sim.set_drive_filter(drive_filter)

                # torque_delay = cassie_env.sim.get_torque_delay()
                # delay_env.sim.set_torque_delay(torque_delay)

                mj_count = 0
            
            # delay_env.sim.copy(cassie_env.sim)
            delay_env.phase = cassie_env.phase
            # delay_env.sim = cassie_env.sim.duplicate()
            # print("cassie env step")
            # cassie_env.step_sim_basic(action)
            real_s = cassie_env.cassie_state

            sim_target = action + cassie_env.offset
            mj_u = pd_in_t()
            for k in range(5):
                mj_u.leftLeg.motorPd.pGain[k]  = P[k]
                mj_u.rightLeg.motorPd.pGain[k] = P[k]

                mj_u.leftLeg.motorPd.dGain[k]  = D[k]
                mj_u.rightLeg.motorPd.dGain[k] = D[k]

                mj_u.leftLeg.motorPd.torque[k]  = 0 # Feedforward torque
                mj_u.rightLeg.motorPd.torque[k] = 0 

                mj_u.leftLeg.motorPd.pTarget[k]  = sim_target[k]#action[k] + cassie_env.offset[k]
                mj_u.rightLeg.motorPd.pTarget[k] = sim_target[k+5]#action[k + 5] + cassie_env.offset[k + 5]

                mj_u.leftLeg.motorPd.dTarget[k]  = 0
                mj_u.rightLeg.motorPd.dTarget[k] = 0
            cassie_env.cassie_state = cassie_env.sim.step_pd(mj_u)
            # exit()
            
            # mj_s = mj_sim.step_pd(mj_u)
            delay_env.cassie_state = delay_env.sim.step_pd(mj_u)
            # print("delay env step")
            # delay_env.step_sim_basic(action)
            # delay_env.sim.copy_state_est(cassie_env.sim)
            # delay_env.sim.copy(cassie_env.sim)
            mj_s = delay_env.cassie_state
            real_s = cassie_env.cassie_state
            mj_count += 1

            real_qpos = np.array(cassie_env.sim.qpos())
            sim_qpos = np.array(delay_env.sim.qpos())
            # print("qpos diff:", np.linalg.norm(real_qpos - sim_qpos))
            real_state = cassie_env.get_full_state()
            delay_state = delay_env.get_full_state()
            ### Can uncomment to see that the state estimator immediately differs after calling step_pd
            # print("state diff:", real_state-delay_state)
            # print(np.array(real_s.pelvis.position[:]) - np.array(mj_s.pelvis.position[:]))
            # exit()

            pel_pos[j*simrate + i, :] = np.array(real_s.pelvis.position[:]) - np.array(mj_s.pelvis.position[:])
            pel_orient[j*simrate + i, :] = np.array(real_s.pelvis.orientation[:]) - np.array(mj_s.pelvis.orientation[:])
            pel_trans_vel[j*simrate + i, :] = np.array(real_s.pelvis.translationalVelocity[:]) - np.array(mj_s.pelvis.translationalVelocity[:])
            pel_rot_vel[j*simrate + i, :] = np.array(real_s.pelvis.rotationalVelocity[:]) - np.array(mj_s.pelvis.rotationalVelocity[:])
            pel_accel[j*simrate + i, :] = np.array(real_s.pelvis.translationalAcceleration[:]) - np.array(mj_s.pelvis.translationalAcceleration[:])

            motors_pos[j*simrate + i, :] = np.array(real_s.motor.position[:]) - np.array(mj_s.motor.position[:])
            motors_vel[j*simrate + i, :] = np.array(real_s.motor.velocity[:]) - np.array(mj_s.motor.velocity[:])
            joints_pos[j*simrate + i, :] = np.array(real_s.joint.position[:]) - np.array(mj_s.joint.position[:])
            joints_vel[j*simrate + i, :] = np.array(real_s.joint.velocity[:]) - np.array(mj_s.joint.velocity[:])
            torques[j*simrate + i, :] = np.array(real_s.motor.torque[:]) - np.array(mj_s.motor.torque[:])

            l_foot_pos[j*simrate + i, :] = np.array(real_s.leftFoot.position[:]) - np.array(mj_s.leftFoot.position[:])
            l_foot_orient[j*simrate + i, :] = np.array(real_s.leftFoot.orientation[:]) - np.array(mj_s.leftFoot.orientation[:])
            l_foot_trans_vel[j*simrate + i, :] = np.array(real_s.leftFoot.footTranslationalVelocity[:]) - np.array(mj_s.leftFoot.footTranslationalVelocity[:])
            l_foot_rot_vel[j*simrate + i, :] = np.array(real_s.leftFoot.footRotationalVelocity[:]) - np.array(mj_s.leftFoot.footRotationalVelocity[:])
            l_foot_toe_force[j*simrate + i, :] = np.array(real_s.leftFoot.toeForce[:]) - np.array(mj_s.leftFoot.toeForce[:])
            l_foot_heel_force[j*simrate + i, :] = np.array(real_s.leftFoot.heelForce[:]) - np.array(mj_s.leftFoot.heelForce[:])
            
            r_foot_pos[j*simrate + i, :] = np.array(real_s.rightFoot.position[:]) - np.array(mj_s.rightFoot.position[:])
            r_foot_orient[j*simrate + i, :] = np.array(real_s.rightFoot.orientation[:]) - np.array(mj_s.rightFoot.orientation[:])
            r_foot_trans_vel[j*simrate + i, :] = np.array(real_s.rightFoot.footTranslationalVelocity[:]) - np.array(mj_s.rightFoot.footTranslationalVelocity[:])
            r_foot_rot_vel[j*simrate + i, :] = np.array(real_s.rightFoot.footRotationalVelocity[:]) - np.array(mj_s.rightFoot.footRotationalVelocity[:])
            r_foot_toe_force[j*simrate + i, :] = np.array(real_s.rightFoot.toeForce[:]) - np.array(mj_s.rightFoot.toeForce[:])
            r_foot_heel_force[j*simrate + i, :] = np.array(real_s.rightFoot.heelForce[:]) - np.array(mj_s.rightFoot.heelForce[:])

        cassie_env.phase += 1
        if cassie_env.phase >= cassie_env.phaselen:
            cassie_env.phase -= cassie_env.phaselen

    SAVE_NAME = './mj_diffdata_sync{}.npz'.format(sync)
    np.savez(SAVE_NAME, pel_pos=pel_pos, pel_orient=pel_orient, pel_trans_vel=pel_trans_vel, pel_rot_vel=pel_rot_vel, pel_accel=pel_accel, 
            motors_pos=motors_pos, motors_vel = motors_vel, joints_pos = joints_pos, joints_vel=joints_vel, torques = torques, 
            l_foot_pos = l_foot_pos, l_foot_orient = l_foot_orient, l_foot_trans_vel = l_foot_trans_vel, l_foot_rot_vel = l_foot_rot_vel, l_foot_toe_force=l_foot_toe_force, l_foot_heel_force=l_foot_heel_force,
            r_foot_pos = r_foot_pos, r_foot_orient = r_foot_orient, r_foot_trans_vel = r_foot_trans_vel, r_foot_rot_vel = r_foot_rot_vel, r_foot_toe_force=r_foot_toe_force, r_foot_heel_force=r_foot_heel_force)


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="./trained_models/ppo/Cassie-v0/7b7e24-seed0/", help="path to folder containing policy and run details")
parser.add_argument("--plot", default=False, action="store_true", help="Whether to plot or save data")
parser.add_argument("--save", default=False, action="store_true", help="Whether to save or show plots")
parser.add_argument("--sync", type=int, default=1, help="How often to sync up the mujoco state with real logged state")

args = parser.parse_args()

run_args = pickle.load(open(args.path + "experiment.pkl", "rb"))

if not hasattr(run_args, "simrate"):
    run_args.simrate = 50
    print("manually choosing simrate as 50 (40 Hz)")
if not hasattr(run_args, "phase_based"):
    run_args.phase_based = False

policy = torch.load(args.path + "actor.pt")
policy.eval()

env_fn = env_factory(run_args.env_name, traj=run_args.traj, simrate=run_args.simrate, phase_based=run_args.phase_based, clock_based=run_args.clock_based, state_est=run_args.state_est, no_delta=run_args.no_delta, learn_gains=run_args.learn_gains, ik_baseline=run_args.ik_baseline, dynamics_randomization=run_args.dyn_random, mirror=run_args.mirror, reward=run_args.reward, history=run_args.history)
env = env_fn().env

if args.plot:
    # if not os.path.exists(os.path.join(args.path, "diffdata_sync{}.npz".format(args.sync))):
    #     print("Have not processed data with this sync yet, processing now.")
    #     process_data(args.path, args.sync)
    plot_data(args.sync, args.save)

else:
    process_data(policy, env, env_fn().env, args.sync, 60)