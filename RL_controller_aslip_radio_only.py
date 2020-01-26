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
        self.speed = self.speeds[5]
        self.trajectory = self.trajectories[5]

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
target_log = [] # PD target log
traj_log   = [] # reference trajectory log

speed_log  = [] # speed input to reference trajectory library log

filename = "test.p"
filep = open(filename, "wb")

max_speed = 2.0
min_speed = 0.0

if len(sys.argv) > 1:
    filename = PREFIX + "logs/" + sys.argv[1]
else:
    filename = PREFIX + "logs/" + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M')

def log(sto="final"):
    data = {"time": time_log, "output": output_log, "input": input_log, "state": state_log, "target": target_log, "trajectory": traj_log, "speed": speed_log}

    filep = open(filename + "_log" + str(sto) + ".pkl", "wb")

    pickle.dump(data, filep)

    filep.close()

atexit.register(log)

# Prevent latency issues by disabling multithreading in pytorch
torch.set_num_threads(1)

policy = torch.load("./trained_models/aslip_unified_task0_v7.pt")
policy.eval()

# Prevent latency issues by disabling multithreading in pytorch
torch.set_num_threads(1)

max_speed = 2.0
min_speed = 0.0
max_y_speed = 0.0
min_y_speed = 0.0

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
_, offset = traj.update_info(min_speed)
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

orient_add = 0
old_settings = termios.tcgetattr(sys.stdin)

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

    # Radio control
    orient_add -= state.radio.channel[3] / 60.0

    if state.radio.channel[8] < 1:
        orient_add = quaternion2euler(state.pelvis.orientation[:])[2]

    print("orient add: ", orient_add)
    traj.speed = max(min_speed, state.radio.channel[0] * max_speed)
    traj.speed = min(max_speed, state.radio.channel[0] * max_speed)
    print("speed input: ", state.radio.channel[0] * max_speed)
    # traj.phase_add = state.radio.channel[5] + 1
    # env.y_speed = max(min_y_speed, -state.radio.channel[1] * max_y_speed)
    # env.y_speed = min(max_y_speed, -state.radio.channel[1] * max_y_speed)

    traj.update_info(traj.speed)

    clock = [np.sin(2 * np.pi *  traj.phase * traj.freq_adjust / traj.phaselen), np.cos(2 * np.pi *  traj.phase * traj.freq_adjust / traj.phaselen)]



    # euler_orient = quaternion2euler(state.pelvis.orientation[:]) 
    # print("euler orient: ", euler_orient + np.array([orient_add, 0, 0]))
    # new_orient = euler2quat(euler_orient + np.array([orient_add, 0, 0]))
    quaternion = euler2quat(z=orient_add, y=0, x=0)
    iquaternion = inverse_quaternion(quaternion)
    new_orient = quaternion_product(iquaternion, state.pelvis.orientation[:])

    new_translationalVelocity = rotate_by_quaternion(state.pelvis.translationalVelocity[:], iquaternion)


    print("orig orientation: ", state.pelvis.orientation[:])
    print('new_orientation: {}'.format(new_orient))
        
    # ext_state = np.concatenate((clock, [speed, y_speed]))
    ext_state = np.concatenate((clock, [traj.speed] ))
    robot_state = np.concatenate([
            [state.pelvis.position[2] - state.terrain.height], # pelvis height
            new_orient,
            # state.pelvis.orientation[:],                                 # pelvis orientation
            state.motor.position[:],                                     # actuated joint positions

            # state.pelvis.translationalVelocity[:],                       # pelvis translational velocity
            new_translationalVelocity[:],
            state.pelvis.rotationalVelocity[:],                          # pelvis rotational velocity 
            state.motor.velocity[:],                                     # actuated joint velocities

            state.pelvis.translationalAcceleration[:],                   # pelvis translational acceleration
            
            state.joint.position[:],                                     # unactuated joint positions
            state.joint.velocity[:]                                      # unactuated joint velocities
    ])
    RL_state = np.concatenate([robot_state, ext_state])

    actual_speed = state.pelvis.translationalVelocity[0]
    print("target speed: {:.2f}\tactual speed: {:.2f}\tfreq: {}".format(traj.speed, actual_speed, traj.freq_adjust))

    #pretending the height is always 1.0
    # RL_state[0] = 1.0
    
    # Construct input vector
    torch_state = torch.Tensor(RL_state)
    # torch_state = shared_obs_stats.normalize(torch_state)

    # Get action
    action = policy.act(torch_state, True)
    env_action = action.data.numpy()
    target = env_action + traj.offset

    # print(state.pelvis.position[2] - state.terrain.height)

    # Send action
    for i in range(5):
        u.leftLeg.motorPd.pTarget[i] = target[i]
        u.rightLeg.motorPd.pTarget[i] = target[i+5]
    #time.sleep(0.005)
    cassie.send_pd(u)

    # Measure delay
    print('delay: {:6.1f} ms'.format((time.monotonic() - t) * 1000))

    # Logging
    time_log.append(time.time())
    state_log.append(state)
    input_log.append(RL_state)
    output_log.append(env_action)
    target_log.append(target)
    speed_log.append(traj.speed)

    # Track phase
    traj.phase += traj.phase_add
    if traj.phase >= traj.phaselen:
        traj.phase = 0
        traj.counter += 1