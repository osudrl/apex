from cassie.cassiemujoco.cassieUDP import *
from cassie.cassiemujoco.cassiemujoco_ctypes import *
from cassie.trajectory import CassieTrajectory
# from cassie.speed_env import CassieEnv_speed
# from cassie.speed_double_freq_env import CassieEnv_speed_dfreq
# from cassie.speed_no_delta_env import CassieEnv_speed_no_delta
# from cassie.speed_no_delta_neutral_foot_env import CassieEnv_speed_no_delta_neutral_foot

import time
import numpy as np
import torch
import pickle
from rl.policies import GaussianMLP
import platform
from quaternion_function import *

#import signal 
import atexit


"""
We need to include the trajectory library for the right offset information, as well as the right phaselen and speed
"""

def getAllTrajectories(speeds):
    trajectories = []

    for i, speed in enumerate(speeds):
        dirname = os.path.dirname(__file__)
        traj_path = os.path.join(dirname, "trajectory", "aslipTrajsImprovedCost_freq_correction", "walkCycle_{}.pkl".format(speed))
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
        self.rfoot = np.copy(trajectory["rfoot"])
        self.lfoot = np.copy(trajectory["lfoot"])

# simrate used to be 60
class TrajectoryInfo:
    def __init__(self):
        
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
        self.prev_action = ref_pos[self.pos_idx]
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

    def update_info(self, new_speed):

        # new speed and trajectory
        self.speed = new_speed
        self.trajectory = self.trajectories[ (np.abs(self.speeds - self.speed)).argmin() ]

        # new offset
        ref_pos, ref_vel = self.get_ref_state(self.phase)
        self.prev_action = ref_pos[self.pos_idx]

        # phaselen
        self.phaselen = self.trajectory.length - 1

traj = TrajectoryInfo()

time_log   = [] # time stamp
input_log  = [] # network inputs
output_log = [] # network outputs 
state_log  = [] # cassie state
target_log = [] #PD target log

filename = "test.p"
filep = open(filename, "wb")

def log():
    data = {"time": time_log, "output": output_log, "input": input_log, "state": state_log, "target": target_log}
    pickle.dump(data, filep)

atexit.register(log)

# Prevent latency issues by disabling multithreading in pytorch
torch.set_num_threads(1)

# Prepare model
# env = CassieEnv_speed_no_delta_neutral_foot("walking", clock_based=True, state_est=True)
# env.reset_for_test()
#
speed = 0

# policy = torch.load("./trained_models/nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2.pt")
policy = torch.load("./trained_models/aslip_unified_freq_correction.pt")
policy.eval()

max_speed = 3.0
# min_speed = -1.0
min_speed = 0.0
max_y_speed = 0.0
min_y_speed = 0.0
symmetry = True

# Initialize control structure with gains
P = np.array([100, 100, 88, 96, 50, 100, 100, 88, 96, 50])
D = np.array([10.0, 10.0, 8.0, 9.6, 5.0, 10.0, 10.0, 8.0, 9.6, 5.0])
u = pd_in_t()
for i in range(5):
    u.leftLeg.motorPd.pGain[i] = P[i]
    u.leftLeg.motorPd.dGain[i] = D[i]
    u.rightLeg.motorPd.pGain[i] = P[i+5]
    u.rightLeg.motorPd.dGain[i] = D[i+5]

pos_index = np.array([2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])
pos_mirror_index = np.array([2,3,4,5,6,21,22,23,28,29,30,34,7,8,9,14,15,16,20])
vel_mirror_index = np.array([0,1,2,3,4,5,19,20,21,25,26,27,31,6,7,8,12,13,14,18])
offset = traj.prev_action
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
        curr_max = max_speed / 2# + (max_speed / 2)*state.radio.channel[4]
        #print("curr_max:", curr_max)
        speed_add = (max_speed / 2) * state.radio.channel[4]
        speed = max(min_speed, state.radio.channel[0] * curr_max + speed_add)
        speed = min(max_speed, state.radio.channel[0] * curr_max + speed_add)
        
        print("speed: ", speed)
        traj.phase_add = 1+state.radio.channel[5]
        # env.y_speed = max(min_y_speed, -state.radio.channel[1] * max_y_speed)
        # env.y_speed = min(max_y_speed, -state.radio.channel[1] * max_y_speed)
    else:
        # Automatically change orientation and speed
        tt = time.monotonic() - t0
        orient_add += 0#math.sin(t / 8) / 400
        #env.speed = 0.2
        speed += 0.001#((math.sin(tt / 2)) * max_speed)
        print("speed: ", speed)
        #if env.phase % 14 == 0:
        #	env.speed = (random.randint(-1, 1)) / 2.0
        # print(env.speed)
        speed = max(min_speed, speed)
        speed = min(max_speed, speed)
        # env.y_speed = (math.sin(tt / 2)) * max_y_speed
        # env.y_speed = max(min_y_speed, env.y_speed)
        # env.y_speed = min(max_y_speed, env.y_speed)

    # if env.phase < 14 or symmetry is False:
    	# quaternion = euler2quat(z=env.orientation, y=0, x=0)
    	# iquaternion = inverse_quaternion(quaternion)
    	# new_orientation = quaternion_product(iquaternion, state.pelvis.orientation[:])
    # 	if new_orientation[0] < 0:
    # 		new_orientation = -new_orientation
    # 	new_translationalVelocity = rotate_by_quaternion(state.pelvis.translationalVelocity[:], iquaternion)

    # 	print('quaternion: {}, new_orientation: {}'.format(quaternion, new_orientation))

    # 	# Construct input vector
    # 	if symmetry:
    # 		cassie_state = np.copy(np.concatenate([[state.pelvis.position[2] - state.terrain.height], new_orientation[:], state.motor.position[:], new_translationalVelocity[:], state.pelvis.rotationalVelocity[:], state.motor.velocity[:], state.pelvis.translationalAcceleration[:], state.joint.position[:], state.joint.velocity[:]]))
    # 	else:
    # 		cassie_state = np.copy(np.concatenate([[state.pelvis.position[2] - state.terrain.height], new_orientation[:], state.motor.position[:], new_translationalVelocity[:], state.pelvis.rotationalVelocity[:], state.motor.velocity[:], state.pelvis.translationalAcceleration[:], state.leftFoot.toeForce[:], state.leftFoot.heelForce[:], state.rightFoot.toeForce[:], state.rightFoot.heelForce[:]]))
    # 	ref_pos, ref_vel = env.get_kin_next_state()
    # 	RL_state = np.concatenate([cassie_state, ref_pos[pos_index], ref_vel[vel_index]])
    # else:
    # 	quaternion = euler2quat(z=env.orientation, y=0, x=0)
    # 	cassie_state = get_mirror_state(state, quaternion)
    # 	ref_pos, ref_vel = env.get_kin_next_state()
    # 	ref_vel[1] = -ref_vel[1]
    # 	euler = quaternion2euler(ref_pos[3:7])
    # 	euler[0] = -euler[0]
    # 	euler[2] = -euler[2]
    # 	ref_pos[3:7] = euler2quat(z=euler[2],y=euler[1],x=euler[0])
    # 	RL_state = np.concatenate([cassie_state, ref_pos[pos_mirror_index], ref_vel[vel_mirror_index]])

    traj.update_info(speed)

    # phaselen = 11 # used to be 27

    clock = [np.sin(2 * np.pi *  traj.phase / traj.phaselen), np.cos(2 * np.pi *  traj.phase / traj.phaselen)]
    # clock = [np.sin(2 * np.pi *  phase / 27 * (2000 / 30)), np.cos(2 * np.pi *  phase / 27 * (2000 / 30))]
    # euler_orient = quaternion2euler(state.pelvis.orientation[:]) 
    # print("euler orient: ", euler_orient + np.array([orient_add, 0, 0]))
    # new_orient = euler2quat(euler_orient + np.array([orient_add, 0, 0]))
    quaternion = euler2quat(z=orient_add, y=0, x=0)
    iquaternion = inverse_quaternion(quaternion)
    new_orient = quaternion_product(iquaternion, state.pelvis.orientation[:])
    if new_orient[0] < 0:
        new_orient = -new_orient
    new_translationalVelocity = rotate_by_quaternion(state.pelvis.translationalVelocity[:], iquaternion)
    print('new_orientation: {}'.format(new_orient))
    print('clock: {}'.format(clock))
        
    # ext_state = np.concatenate((clock, [speed]))
    ext_state = clock
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

    
    #pretending the height is always 1.0
    # RL_state[0] = 1.0
    
    # Construct input vector
    torch_state = torch.Tensor(RL_state)
    # torch_state = shared_obs_stats.normalize(torch_state)

    # Get action
    action = policy.act(torch_state, True)
    env_action = action.data.numpy()
    target = env_action + offset

    #print(state.pelvis.position[2] - state.terrain.height)

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

    # Track phase
    
    traj.phase += traj.phase_add
    if traj.phase >= traj.phaselen + 1:
    # if phase >= 28 * (2000 / 30):
        traj.phase = 0
        traj.counter += 1
