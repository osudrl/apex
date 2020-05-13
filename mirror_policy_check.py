import torch
import hashlib, os, pickle
import sys, time
from cassie.quaternion_function import *
import tty
import termios
import select
import numpy as np
from functools import partial
from rl.envs.wrappers import SymmetricEnv
from cassie import CassieEnv, CassiePlayground, CassieStandingEnv, CassieEnv_noaccel_footdist_omniscient, CassieEnv_noaccel_footdist

def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

env = CassieEnv(state_est=True, dynamics_randomization=False, history=0)
env_fn = partial(CassieEnv, state_est=True, dynamics_randomization=False, history=0)
# env = CassieEnv_noaccel_footdist(state_est=True, dynamics_randomization=False, history=0)
# env_fn = partial(CassieEnv_noaccel_footdist, state_est=True, dynamics_randomization=False, history=0)

sym_env = SymmetricEnv(env_fn, mirrored_obs=env_fn().mirrored_obs, mirrored_act=[-5, -6, 7, 8, 9, -0.1, -1, 2, 3, 4])
# obs = env.get_full_state()
# print("obs len: ", len(obs))
# exit()

path = "./trained_models/nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2/"
# path = "./logs/footdist/CassieNoaccelFootDist/noaccel_footdist_speedmatch_seed10/"
policy = torch.load(path + "actor.pt")
policy.eval()

old_settings = termios.tcgetattr(sys.stdin)

orient_add = 0

env.render()
render_state = True
try:
    tty.setcbreak(sys.stdin.fileno())

    state = env.reset_for_test()
    done = False
    timesteps = 0
    eval_reward = 0
    speed = 0.0

    while render_state:
    
        if isData():
            c = sys.stdin.read(1)
            if c == 'w':
                speed += 0.1
            elif c == 's':
                speed -= 0.1
            elif c == 'j':
                env.phase_add += .1
                print("Increasing frequency to: {:.1f}".format(env.phase_add))
            elif c == 'h':
                env.phase_add -= .1
                print("Decreasing frequency to: {:.1f}".format(env.phase_add))
            elif c == 'l':
                orient_add += .1
                print("Increasing orient_add to: ", orient_add)
            elif c == 'k':
                orient_add -= .1
                print("Decreasing orient_add to: ", orient_add)
            elif c == 'p':
                push = 100
                push_dir = 2
                force_arr = np.zeros(6)
                force_arr[push_dir] = push
                env.sim.apply_force(force_arr)

            env.update_speed(speed)
            print("speed: ", env.speed)
        
        if hasattr(env, 'simrate'):
            start = time.time()

        if (not env.vis.ispaused()):
            # Update Orientation
            quaternion = euler2quat(z=orient_add, y=0, x=0)
            iquaternion = inverse_quaternion(quaternion)

            if env.state_est:
                curr_orient = state[1:5]
                curr_transvel = state[15:18]
                # curr_orient = state[6:10]
                # curr_transvel = state[20:23]
            else:
                curr_orient = state[2:6]
                curr_transvel = state[20:23]
            
            new_orient = quaternion_product(iquaternion, curr_orient)

            if new_orient[0] < 0:
                new_orient = -new_orient

            new_translationalVelocity = rotate_by_quaternion(curr_transvel, iquaternion)
            
            if env.state_est:
                state[1:5] = torch.FloatTensor(new_orient)
                state[15:18] = torch.FloatTensor(new_translationalVelocity)
                # state[6:10] = torch.FloatTensor(new_orient)
                # state[20:23] = torch.FloatTensor(new_translationalVelocity)
                # state[0] = 1      # For use with StateEst. Replicate hack that height is always set to one on hardware.
            else:
                state[2:6] = torch.FloatTensor(new_orient)
                state[20:23] = torch.FloatTensor(new_translationalVelocity)          
                
            state = torch.Tensor(state)
            # Calculate mirror state and mirror action
            with torch.no_grad():
                mirror_state = sym_env.mirror_clock_observation(state.unsqueeze(0), env.clock_inds)[0]
                # Mirror pelvis orientation and velocity
                # mir_quat = inverse_quaternion(mirror_state[1:5])
                # mir_quat[2] *= -1
                # mirror_state[1:5] = torch.Tensor(mir_quat)
                # mirror_state[16] *= -1      # y trans vel
                # mir_rot_vel = -mirror_state[18:21]
                # mir_rot_vel[1] *= -1
                # mirror_state[18:21] = mir_rot_vel
                # mirror_state[32] *= -1      # y trans accel
                mir_action = policy.forward(mirror_state, deterministic=True)
                mir_mir_action = sym_env.mirror_action(mir_action.unsqueeze(0)).detach().numpy()[0]
                action = policy.forward(state, deterministic=True).detach().numpy()
            # print("mirror action diff: ", np.linalg.norm(mir_mir_action - action))
            state, reward, done, _ = env.step(mir_mir_action)
            
            eval_reward += reward
            timesteps += 1

            
        render_state = env.render()
        if hasattr(env, 'simrate'):
            # assume 30hz (hack)
            end = time.time()
            delaytime = max(0, 1000 / 30000 - (end-start))
            time.sleep(delaytime)

    print("Eval reward: ", eval_reward)

finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)