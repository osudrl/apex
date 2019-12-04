from .cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis

from .trajectory import CassieTrajectory

from math import floor

import numpy as np 
import os
import random

import pickle

class CassieEnv_rand_dyn:
    def __init__(self, traj, simrate=60, clock_based=False, state_est=False):
        self.sim = CassieSim("./cassie/cassiemujoco/cassie.xml")
        self.vis = None

        self.clock_based = clock_based
        self.state_est = state_est

        if clock_based:
            self.observation_space = np.zeros(42)
            if self.state_est:
                self.observation_space = np.zeros(48)       # Size for use with state est
        else:
            self.observation_space = np.zeros(80)
            if self.state_est:
                self.observation_space = np.zeros(86)       # Size for use with state est

        self.action_space = np.zeros(10)

        dirname = os.path.dirname(__file__)
        if traj == "walking":
            traj_path = os.path.join(dirname, "trajectory", "stepdata.bin")

        elif traj == "stepping":
            traj_path = os.path.join(dirname, "trajectory", "more-poses-trial.bin")

        self.trajectory = CassieTrajectory(traj_path)

        self.P = np.array([100,  100,  88,  96,  50]) 
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])

        self.u = pd_in_t()

        # TODO: should probably initialize this to current state
        self.cassie_state = state_out_t()

        self.simrate = simrate # simulate X mujoco steps with same pd target
                               # 60 brings simulation from 2000Hz to roughly 30Hz

        self.time    = 0 # number of time steps in current episode
        self.phase   = 0 # portion of the phase the robot is in
        self.counter = 0 # number of phase cycles completed in episode

        # NOTE: a reference trajectory represents ONE phase cycle

        # should be floor(len(traj) / simrate) - 1
        # should be VERY cautious here because wrapping around trajectory
        # badly can cause assymetrical/bad gaits
        self.phaselen = floor(len(self.trajectory) / self.simrate) - 1

        # see include/cassiemujoco.h for meaning of these indices
        self.pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

        self.speed = 0
        # maybe make ref traj only send relevant idxs?
        ref_pos, ref_vel = self.get_ref_state(self.phase)
        self.prev_action = ref_pos[self.pos_idx]
        self.phase_add = 1


        # Record default dynamics parameters
        self.default_damping = self.sim.get_dof_damping()
        self.default_mass = self.sim.get_body_mass()
        self.default_ipos = self.sim.get_body_ipos()
        self.default_fric = self.sim.get_ground_friction()
        #print(self.default_damping)
        #print(self.default_mass)
        #print(self.default_ipos)
        #print(self.default_fric)
        #input()
    

    def step_simulation(self, action):

        # maybe make ref traj only send relevant idxs?
        ref_pos, ref_vel = self.get_ref_state(self.phase + self.phase_add)
        
        target = action + ref_pos[self.pos_idx]
        
        self.u = pd_in_t()
        for i in range(5):
            # TODO: move setting gains out of the loop?
            # maybe write a wrapper for pd_in_t ?
            self.u.leftLeg.motorPd.pGain[i]  = self.P[i]
            self.u.rightLeg.motorPd.pGain[i] = self.P[i]

            self.u.leftLeg.motorPd.dGain[i]  = self.D[i]
            self.u.rightLeg.motorPd.dGain[i] = self.D[i]

            self.u.leftLeg.motorPd.torque[i]  = 0 # Feedforward torque
            self.u.rightLeg.motorPd.torque[i] = 0 

            self.u.leftLeg.motorPd.pTarget[i]  = target[i]
            self.u.rightLeg.motorPd.pTarget[i] = target[i + 5]

            self.u.leftLeg.motorPd.dTarget[i]  = 0
            self.u.rightLeg.motorPd.dTarget[i] = 0

        self.cassie_state = self.sim.step_pd(self.u)

    def step(self, action):
        for _ in range(self.simrate):
            self.step_simulation(action)

        height = self.sim.qpos()[2]

        self.time  += 1
        self.phase += self.phase_add

        if self.phase > self.phaselen:
            self.phase = 0
            self.counter += 1

        # Early termination
        done = not(height > 0.4 and height < 3.0)

        reward = self.compute_reward()

        # TODO: make 0.3 a variable/more transparent
        if reward < 0.3:
            done = True

        return self.get_full_state(), reward, done, {}

    def reset(self, randomize=True):

        # Randomize dynamics:
        if randomize:
            damp = self.default_damping
            weak_factor = 1
            strong_factor = 1
            pelvis_damp_range   = [[damp[0], damp[0]], [damp[1], damp[1]], [damp[2], damp[2]], [damp[3], damp[3]], [damp[4], damp[4]], [damp[5], damp[5]]]                 # 0->5

            hip_damp_range      = [[damp[6]/weak_factor, damp[6]*weak_factor], [damp[7]/weak_factor,  damp[7]*weak_factor],  [damp[8]/weak_factor,  damp[8]*weak_factor]]  # 6->8 and 19->21
            achilles_damp_range = [[damp[9]/weak_factor, damp[9]*weak_factor], [damp[10]/weak_factor, damp[10]*weak_factor], [damp[11]/weak_factor, damp[11]*weak_factor]] # 9->11 and 22->24

            knee_damp_range     = [[damp[12]/weak_factor, damp[12]*weak_factor]]   # 12 and 25
            shin_damp_range     = [[damp[13]/weak_factor, damp[13]*weak_factor]]   # 13 and 26
            tarsus_damp_range   = [[damp[14], damp[14]*strong_factor]]             # 14 and 27
            heel_damp_range     = [[damp[15], damp[15]]]                           # 15 and 28
            fcrank_damp_range   = [[damp[16]/weak_factor, damp[16]*weak_factor]]   # 16 and 29
            prod_damp_range     = [[damp[17], damp[17]]]                           # 17 and 30
            foot_damp_range     = [[damp[18]/weak_factor, damp[18]*weak_factor]]   # 18 and 31

            side_damp = hip_damp_range + achilles_damp_range + knee_damp_range + shin_damp_range + tarsus_damp_range + heel_damp_range + fcrank_damp_range + prod_damp_range + foot_damp_range
            damp_range = pelvis_damp_range + side_damp + side_damp
            damp_noise = [np.random.uniform(a, b) for a, b in damp_range]

            #nbody layout:
            # 0:  worldbody (zero)
            # 1:  pelvis

            # 2:  left hip roll 
            # 3:  left hip yaw
            # 4:  left hip pitch
            # 5:  left achilles rod
            # 6:  left knee
            # 7:  left knee spring
            # 8:  left shin
            # 9:  left tarsus
            # 10:  left heel spring
            # 12:  left foot crank
            # 12: left plantar rod
            # 13: left foot

            # 14: right hip roll 
            # 15: right hip yaw
            # 16: right hip pitch
            # 17: right achilles rod
            # 18: right knee
            # 19: right knee spring
            # 20: right shin
            # 21: right tarsus
            # 22: right heel spring
            # 23: right foot crank
            # 24: right plantar rod
            # 25: right foot

            hi = 1.2
            lo = 0.8
            m = self.default_mass
            pelvis_mass_range      = [[lo*m[1],  hi*m[1]]]  # 1
            hip_mass_range         = [[lo*m[2],  hi*m[2]],  # 2->4 and 14->16
                                      [lo*m[3],  hi*m[3]], 
                                      [lo*m[4],  hi*m[4]]] 

            achilles_mass_range    = [[lo*m[5],  hi*m[5]]]  # 5 and 17
            knee_mass_range        = [[lo*m[6],  hi*m[6]]]  # 6 and 18
            knee_spring_mass_range = [[lo*m[7],  hi*m[7]]]  # 7 and 19
            shin_mass_range        = [[lo*m[8],  hi*m[8]]]  # 8 and 20
            tarsus_mass_range      = [[lo*m[9],  hi*m[9]]]  # 9 and 21
            heel_spring_mass_range = [[lo*m[10], hi*m[10]]] # 10 and 22
            fcrank_mass_range      = [[lo*m[11], hi*m[11]]] # 11 and 23
            prod_mass_range        = [[lo*m[12], hi*m[12]]] # 12 and 24
            foot_mass_range        = [[lo*m[13], hi*m[13]]] # 13 and 25

            side_mass = hip_mass_range + achilles_mass_range + knee_mass_range + knee_spring_mass_range + shin_mass_range + tarsus_mass_range + heel_spring_mass_range + fcrank_mass_range + prod_mass_range + foot_mass_range
            mass_range = [[0, 0]] + pelvis_mass_range + side_mass + side_mass
            mass_noise = [np.random.uniform(a, b) for a, b in mass_range]

            delta = 0.001
            com_noise = [0, 0, 0] + [self.default_ipos[i] + np.random.uniform(-delta, delta) for i in range(3, len(self.default_ipos))]

            """
            pelvis_com_range       = [[0.05066, 0.05066], [0.000346, 0.000346], [0.02841, 0.02841]]   # 3->5

            left_hip_com_range     = [[-0.01793, -0.01793], [0.0001, 0.0001], [-0.04428, -0.04428], [0.0, 0.0], [-1e-5, -1e-5], [-0.034277, -0.034277], [0.05946, 0.05946], [0.00005, 0.00005], [-0.03581, -0.03581]] # 6->14
            right_hip_com_range    = [[-0.01793, -0.01793], [0.0001, 0.0001], [-0.04428, -0.04428], [0.0, 0.0], [ 1e-5,  1e-5], [-0.034277, -0.034277], [0.05946, 0.05946], [0.00005, 0.00005], [ 0.03581,  0.03581]] # 42->50

            achilles_com_range     = [[0.24719, 0.24719], [0.0, 0.0], [0.0, 0.0]]                         # 15->17 and 51->53

            left_knee_com_range    = [[0.023, 0.023], [0.03207, 0.03207], [-0.002181, -0.002181]]         # 18->20
            right_knee_com_range   = [[0.023, 0.023], [0.03207, 0.03207], [ 0.002181,  0.002181]]         # 54->56

            knee_spring_com_range  = [[0.0836, 0.0836], [0.0034, 0.0034], [0.0, 0.0]]                     # 21->23 and 57->59

            left_shin_com_range    = [[0.18338, 0.18338], [0.001169, 0.001169], [ 0.0002123,  0.0002123]] # 24->26
            right_shin_com_range   = [[0.18338, 0.18338], [0.001169, 0.001169], [-0.0002123, -0.0002123]] # 60->62

            left_tarsus_com_range  = [[0.11046, 0.11046], [-0.03058, -0.03058], [-0.00131, -0.00131]]     # 27->29
            right_tarsus_com_range = [[0.11046, 0.11046], [-0.03058, -0.03058], [ 0.00131,  0.00131]]     # 63->65

            heel_com_range         = [[0.081, 0.081], [0.0022, 0.0022], [0.0, 0.0]]                       # 30->32 and 66->68

            left_fcrank_com_range  = [[0.00493, 0.00493], [0.00002, 0.00002], [-0.00215, -0.00215]]       # 33->35 and 69->71
            right_fcrank_com_range = [[0.00493, 0.00493], [0.00002, 0.00002], [ 0.00215,  0.00215]]       # 33->35 and 69->71

            prod_com_range         = [[0.17792, 0.17792], [0.0, 0.0], [0.0, 0.0]]                         # 36->38 and 72->74

            left_foot_com_range    = [[0.00474, 0.00474], [0.02748, 0.02748], [-0.00014, -0.00014]]       # 39->41 and 75->77
            right_foot_com_range   = [[0.00474, 0.00474], [0.02748, 0.02748], [ 0.00014,  0.00014]]       # 39->41 and 75->77

            left_com  = left_hip_com_range  + achilles_com_range + left_knee_com_range  + knee_spring_com_range + left_shin_com_range  + left_tarsus_com_range  + heel_com_range + left_fcrank_com_range  + prod_com_range + left_foot_com_range
            right_com = right_hip_com_range + achilles_com_range + right_knee_com_range + knee_spring_com_range + right_shin_com_range + right_tarsus_com_range + heel_com_range + right_fcrank_com_range + prod_com_range + right_foot_com_range

            com_range = [[0, 0], [0, 0], [0, 0]] + pelvis_com_range + left_com + right_com
            com_noise = [np.random.uniform(a, b) for a, b in com_range]
            """

            fric_noise = [np.random.uniform(0.3, 1.3)] + list(self.default_fric[1:])

            self.sim.set_dof_damping(np.clip(damp_noise, 0, None))
            self.sim.set_body_mass(np.clip(mass_noise, 0, None))
            self.sim.set_body_ipos(np.clip(com_noise, 0, None))
            self.sim.set_ground_friction(np.clip(fric_noise, 0, None))

        self.phase = random.randint(0, self.phaselen)
        self.time = 0
        self.counter = 0

        qpos, qvel = self.get_ref_state(self.phase)

        self.sim.set_qpos(qpos)
        self.sim.set_qvel(qvel)

        # Need to reset u? Or better way to reset cassie_state than taking step
        self.cassie_state = self.sim.step_pd(self.u)

        self.speed = (random.randint(0, 10)) / 10
        # maybe make ref traj only send relevant idxs?
        ref_pos, ref_vel = self.get_ref_state(self.phase)
        self.prev_action = ref_pos[self.pos_idx]

        return self.get_full_state()

    # used for plotting against the reference trajectory
    def reset_for_test(self):
        self.phase = 0
        self.time = 0
        self.counter = 0

        qpos, qvel = self.get_ref_state(self.phase)

        self.sim.set_qpos(qpos)
        self.sim.set_qvel(qvel)

        # maybe make ref traj only send relevant idxs?
        ref_pos, ref_vel = self.get_ref_state(self.phase)
        self.prev_action = ref_pos[self.pos_idx]

        # Need to reset u? Or better way to reset cassie_state than taking step
        self.cassie_state = self.sim.step_pd(self.u)

        return self.get_full_state()
    
    def set_joint_pos(self, jpos, fbpos=None, iters=5000):
        """
        Kind of hackish. 
        This takes a floating base position and some joint positions
        and abuses the MuJoCo solver to get the constrained forward
        kinematics. 

        There might be a better way to do this, e.g. using mj_kinematics
        """

        # actuated joint indices
        joint_idx = [7, 8, 9, 14, 20,
                     21, 22, 23, 28, 34]

        # floating base indices
        fb_idx = [0, 1, 2, 3, 4, 5, 6]

        for _ in range(iters):
            qpos = np.copy(self.sim.qpos())
            qvel = np.copy(self.sim.qvel())

            qpos[joint_idx] = jpos

            if fbpos is not None:
                qpos[fb_idx] = fbpos

            self.sim.set_qpos(qpos)
            self.sim.set_qvel(0 * qvel)

            self.sim.step_pd(pd_in_t())


    # NOTE: this reward is slightly different from the one in Xie et al
    # see notes for details
    def compute_reward(self):
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel())

        ref_pos, ref_vel = self.get_ref_state(self.phase)

        # TODO: should be variable; where do these come from?
        # TODO: see magnitude of state variables to gauge contribution to reward
        weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]

        joint_error       = 0
        com_error         = 0
        orientation_error = 0
        spring_error      = 0

        # each joint pos
        for i, j in enumerate(self.pos_idx):
            target = ref_pos[j]
            actual = qpos[j]

            joint_error += 30 * weight[i] * (target - actual) ** 2

        # center of mass: x, y, z
        for j in [0, 1, 2]:
            target = ref_pos[j]
            actual = qpos[j]

            # NOTE: in Xie et al y target is 0

            com_error += (target - actual) ** 2
        
        # COM orientation: qx, qy, qz
        for j in [4, 5, 6]:
            target = ref_pos[j] # NOTE: in Xie et al orientation target is 0
            actual = qpos[j]

            orientation_error += (target - actual) ** 2

        # left and right shin springs
        for i in [15, 29]:
            target = ref_pos[i] # NOTE: in Xie et al spring target is 0
            actual = qpos[i]

            spring_error += 1000 * (target - actual) ** 2      
        
        reward = 0.5 * np.exp(-joint_error) +       \
                 0.3 * np.exp(-com_error) +         \
                 0.1 * np.exp(-orientation_error) + \
                 0.1 * np.exp(-spring_error)

        # reward = np.sign(qvel[0])*qvel[0]**2
        # desired_speed = 3.0
        # speed_diff = np.abs(qvel[0] - desired_speed)
        # if speed_diff > 1:
        #     speed_diff = speed_diff**2
        # reward = 20 - speed_diff

        return reward

    # get the corresponding state from the reference trajectory for the current phase
    def get_ref_state(self, phase=None):
        if phase is None:
            phase = self.phase

        if phase > self.phaselen:
            phase = 0

        pos = np.copy(self.trajectory.qpos[phase * self.simrate])

        # this is just setting the x to where it "should" be given the number
        # of cycles
        # pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * self.counter
        
        # ^ should only matter for COM error calculation,
        # gets dropped out of state variable for input reasons

        ###### Setting variable speed  #########
        pos[0] *= self.speed
        pos[0] += (self.trajectory.qpos[-1, 0]- self.trajectory.qpos[0, 0])* self.counter * self.speed
        ######                          ########

        # setting lateral distance target to 0?
        # regardless of reference trajectory?
        pos[1] = 0

        vel = np.copy(self.trajectory.qvel[phase * self.simrate])
        vel[0] *= self.speed

        return pos, vel

    def get_full_state(self):
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel()) 

        ref_pos, ref_vel = self.get_ref_state(self.phase + self.phase_add)

        # TODO: maybe convert to set subtraction for clarity
        # {i for i in range(35)} - 
        # {0, 10, 11, 12, 13, 17, 18, 19, 24, 25, 26, 27, 31, 32, 33}

        # this is everything except pelvis x and qw, achilles rod quaternions, 
        # and heel spring/foot crank/plantar rod angles
        # note: x is forward dist, y is lateral dist, z is height

        # makes sense to always exclude x because it is in global coordinates and
        # irrelevant to phase-based control. Z is inherently invariant to
        # trajectory despite being global coord. Y is only invariant to straight
        # line trajectories.

        # [ 0] Pelvis y
        # [ 1] Pelvis z
        # [ 2] Pelvis orientation qw
        # [ 3] Pelvis orientation qx
        # [ 4] Pelvis orientation qy
        # [ 5] Pelvis orientation qz
        # [ 6] Left hip roll         (Motor [0])
        # [ 7] Left hip yaw          (Motor [1])
        # [ 8] Left hip pitch        (Motor [2])
        # [ 9] Left knee             (Motor [3])
        # [10] Left shin                        (Joint [0])
        # [11] Left tarsus                      (Joint [1])
        # [12] Left foot             (Motor [4], Joint [2])
        # [13] Right hip roll        (Motor [5])
        # [14] Right hip yaw         (Motor [6])
        # [15] Right hip pitch       (Motor [7])
        # [16] Right knee            (Motor [8])
        # [17] Right shin                       (Joint [3])
        # [18] Right tarsus                     (Joint [4])
        # [19] Right foot            (Motor [9], Joint [5])
        pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])

        # [ 0] Pelvis x
        # [ 1] Pelvis y
        # [ 2] Pelvis z
        # [ 3] Pelvis orientation wx
        # [ 4] Pelvis orientation wy
        # [ 5] Pelvis orientation wz
        # [ 6] Left hip roll         (Motor [0])
        # [ 7] Left hip yaw          (Motor [1])
        # [ 8] Left hip pitch        (Motor [2])
        # [ 9] Left knee             (Motor [3])
        # [10] Left shin                        (Joint [0])
        # [11] Left tarsus                      (Joint [1])
        # [12] Left foot             (Motor [4], Joint [2])
        # [13] Right hip roll        (Motor [5])
        # [14] Right hip yaw         (Motor [6])
        # [15] Right hip pitch       (Motor [7])
        # [16] Right knee            (Motor [8])
        # [17] Right shin                       (Joint [3])
        # [18] Right tarsus                     (Joint [4])
        # [19] Right foot            (Motor [9], Joint [5])
        vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])

        if self.clock_based:
            #qpos[self.pos_idx] -= ref_pos[self.pos_idx]
            #qvel[self.vel_idx] -= ref_vel[self.vel_idx]

            clock = [np.sin(2 * np.pi *  self.phase / self.phaselen),
                     np.cos(2 * np.pi *  self.phase / self.phaselen)]
            
            ext_state = clock

        else:
            ext_state = np.concatenate([ref_pos[pos_index], ref_vel[vel_index]])

        # Use state estimator
        robot_state = np.concatenate([
            [self.cassie_state.pelvis.position[2] - self.cassie_state.terrain.height], # pelvis height
            self.cassie_state.pelvis.orientation[:],                                 # pelvis orientation
            self.cassie_state.motor.position[:],                                     # actuated joint positions

            self.cassie_state.pelvis.translationalVelocity[:],                       # pelvis translational velocity
            self.cassie_state.pelvis.rotationalVelocity[:],                          # pelvis rotational velocity 
            self.cassie_state.motor.velocity[:],                                     # actuated joint velocities

            self.cassie_state.pelvis.translationalAcceleration[:],                   # pelvis translational acceleration
            
            self.cassie_state.joint.position[:],                                     # unactuated joint positions
            self.cassie_state.joint.velocity[:]                                      # unactuated joint velocities
        ])

        if self.state_est:
            return np.concatenate([robot_state,  
                               ext_state])
        else:
            return np.concatenate([qpos[pos_index], 
                               qvel[vel_index], 
                               ext_state])

    def render(self):
        if self.vis is None:
            self.vis = CassieVis(self.sim, "./cassie/cassiemujoco/cassie.xml")

        return self.vis.draw(self.sim)

