from cassie.cassiemujoco.cassieUDP import *
from cassie.cassiemujoco.cassiemujoco_ctypes import *


import time
import numpy as np
import torch
import pickle
import platform
from cassie.quaternion_function import *

#import signal 
import atexit
import sys
import datetime


"""
We need to include the trajectory library for the right offset information, as well as the right phaselen and speed
"""

def getAllTrajectories(speeds):
    trajectories = []

    for i, speed in enumerate(speeds):
        dirname = os.path.dirname(__file__)
        traj_path = os.path.join(dirname, "cassie", "trajectory", "aslipTrajsTaskSpace", "walkCycle_{}.pkl".format(speed))
        trajectories.append(CassieIKTrajectory(traj_path))

    # print("Got all trajectories")
    return trajectories

class CassieIKTrajectory:
    def __init__(self, filepath):
        with open(filepath, "rb") as f:
            trajectory = pickle.load(f)

        self.qpos = np.copy(trajectory["qpos"])
        self.length = self.qpos.shape[0]
        self.qvel = np.copy(trajectory["qvel"])
        self.rpos = np.copy(trajectory["rpos"])
        self.rvel = np.copy(trajectory["rvel"])
        self.lpos = np.copy(trajectory["lpos"])
        self.lvel = np.copy(trajectory["lvel"])
        self.cpos = np.copy(trajectory["cpos"])
        self.cvel = np.copy(trajectory["cvel"])

# simrate used to be 60
class TrajectoryInfo:
    def __init__(self):

        self.freq_adjust = 1
        
        self.speeds = [x / 10 for x in range(0, 21)]
        self.trajectories = getAllTrajectories(self.speeds)
        self.num_speeds = len(self.trajectories)

        self.time    = 0 # number of time steps in current episode
        self.phase   = 0 # portion of the phase the robot is in
        self.counter = 0 # number of phase cycles completed in episode

        # NOTE: each trajectory in trajectories should have the same length
        self.speed = self.speeds[0]
        self.trajectory = self.trajectories[0]

        # NOTE: a reference trajectory represents ONE phase cycle

        # should be floor(len(traj) / simrate) - 1
        # should be VERY cautious here because wrapping around trajectory
        # badly can cause assymetrical/bad gaits
        # self.phaselen = floor(self.trajectory.length / self.simrate) - 1
        self.phaselen = self.trajectory.length - 1

        # see include/cassiemujoco.h for meaning of these indices
        self.pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

        # maybe make ref traj only send relevant idxs?
        ref_pos, ref_vel = self.get_ref_state(self.phase)
        self.offset = ref_pos[self.pos_idx]
        self.phase_add = 1

    # get the corresponding state from the reference trajectory for the current phase
    def get_ref_state(self, phase=None):
        if phase is None:
            phase = self.phase

        if phase > self.phaselen:
            phase = 0

        # pos = np.copy(self.trajectory.qpos[phase * self.simrate])
        pos = np.copy(self.trajectory.qpos[phase])

        # this is just setting the x to where it "should" be given the number
        # of cycles
        #pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * self.counter
        pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * self.counter
        
        # ^ should only matter for COM error calculation,
        # gets dropped out of state variable for input reasons

        # setting lateral distance target to 0?
        # regardless of reference trajectory?
        pos[1] = 0

        # vel = np.copy(self.trajectory.qvel[phase * self.simrate])
        vel = np.copy(self.trajectory.qvel[phase])

        return pos, vel

    def get_ref_ext_state(self, phase=None):

        if phase is None:
            phase = self.phase

        if phase > self.phaselen:
            phase = 0

        rpos = np.copy(self.trajectory.rpos[phase])
        rvel = np.copy(self.trajectory.rvel[phase])
        lpos = np.copy(self.trajectory.lpos[phase])
        lvel = np.copy(self.trajectory.lvel[phase])
        cpos = np.copy(self.trajectory.cpos[phase])
        cvel = np.copy(self.trajectory.cvel[phase])

        return rpos, rvel, lpos, lvel, cpos, cvel

    def update_info(self, new_speed):

        self.speed = new_speed

        # find closest speed in [0.0, 0.1, ... 3.0]. use this to find new trajectory
        self.trajectory = self.trajectories[ (np.abs([speed_i - self.speed for speed_i in self.speeds])).argmin() ]

        # new offset
        ref_pos, ref_vel = self.get_ref_state(self.phase)
        self.offset = ref_pos[self.pos_idx]

        # phaselen
        old_phaselen = self.phaselen
        self.phaselen = self.trajectory.length - 1

        # update phase
        self.phase = int(self.phaselen * self.phase / old_phaselen)

        return self.phaselen, self.offset



time_log   = [] # time stamp
input_log  = [] # network inputs
output_log = [] # network outputs 
state_log  = [] # cassie state
target_log = [] #PD target log
traj_log   = [] # reference trajectory log

simrate = 60

PREFIX = "./"
# PREFIX = "/home/robot/Work/jdao_cassie-rl-testing/"
# PREFIX = "/home/robot/Desktop/Testing/jdao_cassie-rl-testing/" #Dylan's Prefix

if len(sys.argv) > 1:
    filename = PREFIX + "logs/" + sys.argv[1]
else:
    filename = PREFIX + "logs/" + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M')

def log(sto="final"):
    data = {"time": time_log, "output": output_log, "input": input_log, "state": state_log, "target": target_log, "trajectory": traj_log}

    filep = open(filename + "_log" + str(sto) + ".pkl", "wb")

    pickle.dump(data, filep)

    filep.close()

atexit.register(log)

# Prevent latency issues by disabling multithreading in pytorch
torch.set_num_threads(1)

policy = torch.load("./trained_models/aslip_unified_task10_v4.pt")
policy.eval()

max_speed = 2.0
min_speed = 0.50
max_y_speed = 0.0
min_y_speed = 0.0

# Load trajectories
traj = TrajectoryInfo()

# Initialize control structure with gains
P = np.array([100, 100, 88, 96, 50, 100, 100, 88, 96, 50])
D = np.array([10.0, 10.0, 8.0, 9.6, 5.0, 10.0, 10.0, 8.0, 9.6, 5.0])
u = pd_in_t()
for i in range(5):
    u.leftLeg.motorPd.pGain[i] = P[i]
    u.leftLeg.motorPd.dGain[i] = D[i]
    u.rightLeg.motorPd.pGain[i] = P[i+5]
    u.rightLeg.motorPd.dGain[i] = D[i+5]

act_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
pos_index = np.array([1, 2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])
_, offset = traj.update_info(0.0)
# offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])

# Determine whether running in simulation or on the robot
if platform.node() == 'cassie':
    cassie = CassieUdp(remote_addr='10.10.10.3', remote_port='25010',
                       local_addr='10.10.10.100', local_port='25011')
else:
    cassie = CassieUdp() # local testing
    

# Connect to the simulator or robot
print('Connecting...')
y = None
while y is None:
    cassie.send_pd(pd_in_t())
    time.sleep(0.001)
    y = cassie.recv_newest_pd()
received_data = True
print('Connected!\n')

# Record time
t = time.monotonic()
t0 = t

# Whether or not STO has been TOGGLED (i.e. it does not count the initial STO condition)
# STO = True means that STO is ON (i.e. robot is not running) and STO = False means that STO is
# OFF (i.e. robot *is* running)
sto = True
sto_count = 0

orient_add = 0

# We have multiple modes of operation
# 0: Normal operation, walking with policy
# 1: Start up, Standing Pose with variable height (no balance)
# 2: Stop Drop and hopefully not roll, Damping Mode with no P gain
operation_mode = 0
standing_height = 0.7
MAX_HEIGHT = 0.8
MIN_HEIGHT = 0.4
D_mult = 1  # Reaaaaaally bad stability problems if this is pushed higher as a multiplier
            # Might be worth tuning by joint but something else if probably needed

while True:
    # Wait until next cycle time
    while time.monotonic() - t < 60/2000:
        time.sleep(0.001)
    t = time.monotonic()
    tt = time.monotonic() - t0

    # Get newest state
    state = cassie.recv_newest_pd()

    if state is None:
        print('Missed a cycle')
        continue	

    if platform.node() == 'cassie':
        # Radio control
        orient_add -= state.radio.channel[3] / 60.0

        # Reset orientation on STO
        if state.radio.channel[8] < 0:
            orient_add = quaternion2euler(state.pelvis.orientation[:])[2]

            # Save log files after STO toggle (skipping first STO)
            if sto is False:
                log(sto_count)
                sto_count += 1
                sto = True
                # Clear out logs
                time_log   = [] # time stamp
                input_log  = [] # network inputs
                output_log = [] # network outputs 
                state_log  = [] # cassie state
                target_log = [] #PD target log
        else:
            sto = False

        # Switch the operation mode based on the toggle next to STO
        if state.radio.channel[9] < -0.5: # towards operator means damping shutdown mode
            operation_mode = 2
            #D_mult = 5.5 + 4.5* state.radio.channel[7]     # Tune with right side knob 1x-10x (went unstable really fast)
                                                            # Consider using this for some sort of p gain based 

        elif state.radio.channel[9] > 0.5: # away from the operator means that standing pose
            operation_mode = 1
            standing_height = MIN_HEIGHT + (MAX_HEIGHT - MIN_HEIGHT)*0.5*(state.radio.channel[6] + 1)

        else:                               # Middle means normal walking 
            operation_mode = 0
        
        curr_max = max_speed / 2# + (max_speed / 2)*state.radio.channel[4]
        speed_add = (max_speed / 2) * state.radio.channel[4]
        traj.speed = max(min_speed, state.radio.channel[0] * curr_max + speed_add)
        traj.speed = min(max_speed, state.radio.channel[0] * curr_max + speed_add)

        traj.speed = 0.5
        
        print("speed: ", traj.speed)
        # phase_add = 1+state.radio.channel[5]
        # env.y_speed = max(min_y_speed, -state.radio.channel[1] * max_y_speed)
        # env.y_speed = min(max_y_speed, -state.radio.channel[1] * max_y_speed)
    else:
        # Automatically change orientation and speed
        tt = time.monotonic() - t0
        orient_add += math.sin(t / 8) / 400
        #env.speed = 0.2
        speed = ((math.sin(tt / 2)) * max_speed)
        # speed = ((math.sin(tt / 2)) * max_speed)
        print("speed: ", speed)
        #if env.phase % 14 == 0:
        #	env.speed = (random.randint(-1, 1)) / 2.0
        # print(env.speed)
        traj.speed = max(min_speed, speed)
        traj.speed = min(max_speed, speed)
        # env.y_speed = (math.sin(tt / 2)) * max_y_speed
        # env.y_speed = max(min_y_speed, env.y_speed)
        # env.y_speed = min(max_y_speed, env.y_speed)

    #------------------------------- Normal Walking ---------------------------
    if operation_mode == 0:
        
        # Reassign because it might have been changed by the damping mode
        for i in range(5):
            u.leftLeg.motorPd.pGain[i] = P[i]
            u.leftLeg.motorPd.dGain[i] = D[i]
            u.rightLeg.motorPd.pGain[i] = P[i+5]
            u.rightLeg.motorPd.dGain[i] = D[i+5]

        traj.update_info(traj.speed)

        clock = [np.sin(2 * np.pi *  traj.phase * traj.freq_adjust / traj.phaselen), np.cos(2 * np.pi *  traj.phase * traj.freq_adjust / traj.phaselen)]
        quaternion = euler2quat(z=orient_add, y=0, x=0)
        iquaternion = inverse_quaternion(quaternion)
        new_orient = quaternion_product(iquaternion, state.pelvis.orientation[:])
        if new_orient[0] < 0:
            new_orient = -new_orient
        new_translationalVelocity = rotate_by_quaternion(state.pelvis.translationalVelocity[:], iquaternion)
        print('new_orientation: {}'.format(new_orient))

        ref_pos, ref_vel = traj.get_ref_state(traj.phase)
        ext_state = np.concatenate(traj.get_ref_ext_state(traj.phase))
        robot_state = np.concatenate([
                [state.pelvis.position[2] - state.terrain.height], # pelvis height
                new_orient,                                     # pelvis orientation
                state.motor.position[:],                        # actuated joint positions

                new_translationalVelocity[:],                   # pelvis translational velocity
                state.pelvis.rotationalVelocity[:],             # pelvis rotational velocity 
                state.motor.velocity[:],                        # actuated joint velocities

                state.pelvis.translationalAcceleration[:],      # pelvis translational acceleration
                
                state.joint.position[:],                        # unactuated joint positions
                state.joint.velocity[:]                         # unactuated joint velocities
        ])
        RL_state = np.concatenate([robot_state, ext_state])
        
        #pretending the height is always 1.0
        # RL_state[0] = 1.0

        actual_speed = state.pelvis.translationalVelocity[0]
        print("target speed: {:.2f}\tactual speed: {:.2f}\tfreq: {}".format(traj.speed, actual_speed, traj.freq_adjust))

        # Construct input vector
        torch_state = torch.Tensor(RL_state)
        # torch_state = shared_obs_stats.normalize(torch_state)

        # Get action
        action = policy.act(torch_state, True)
        env_action = action.data.numpy()
        # ref_action = ref_pos[act_idx]
        target = env_action + traj.offset

        # Send action
        for i in range(5):
            u.leftLeg.motorPd.pTarget[i] = target[i]
            u.rightLeg.motorPd.pTarget[i] = target[i+5]
        cassie.send_pd(u)

        # Logging
        # if sto == False:
        #     time_log.append(time.time())
        #     state_log.append(state)
        #     input_log.append(RL_state)
        #     output_log.append(env_action)
        #     target_log.append(target)
        #     traj_log.append(traj.offset)
        time_log.append(time.time())
        state_log.append(state)
        input_log.append(RL_state)
        output_log.append(env_action)
        target_log.append(target)
        traj_log.append(traj.offset)
    #------------------------------- Start Up Standing ---------------------------
    elif operation_mode == 1:
        print('Startup Standing. Height = ' + str(standing_height))
        #Do nothing
        # Reassign with new multiplier on damping
        for i in range(5):
            u.leftLeg.motorPd.pGain[i] = 0.0
            u.leftLeg.motorPd.dGain[i] = 0.0
            u.rightLeg.motorPd.pGain[i] = 0.0
            u.rightLeg.motorPd.dGain[i] = 0.0

        # Send action
        for i in range(5):
            u.leftLeg.motorPd.pTarget[i] = 0.0
            u.rightLeg.motorPd.pTarget[i] = 0.0
        cassie.send_pd(u)

    #------------------------------- Shutdown Damping ---------------------------
    elif operation_mode == 2:

        print('Shutdown Damping. Multiplier = ' + str(D_mult))
        # Reassign with new multiplier on damping
        for i in range(5):
            u.leftLeg.motorPd.pGain[i] = 0.0
            u.leftLeg.motorPd.dGain[i] = D_mult*D[i]
            u.rightLeg.motorPd.pGain[i] = 0.0
            u.rightLeg.motorPd.dGain[i] = D_mult*D[i+5]

        # Send action
        for i in range(5):
            u.leftLeg.motorPd.pTarget[i] = 0.0
            u.rightLeg.motorPd.pTarget[i] = 0.0
        cassie.send_pd(u)

    #---------------------------- Other, should not happen -----------------------
    else:
        print('Error, In bad operation_mode with value: ' + str(operation_mode))
    
    # Measure delay
    print('delay: {:6.1f} ms'.format((time.monotonic() - t) * 1000))

    # Track phase
    traj.phase += traj.phase_add
    if traj.phase >= traj.phaselen:
        traj.phase = 0
        traj.counter += 1
