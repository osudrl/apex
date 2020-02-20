# Consolidated Cassie environment.

from .cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis

from .trajectory import CassieTrajectory, getAllTrajectories
from .reward import *

from math import floor

import numpy as np 
import os
import random

import pickle

class CassieEnv_v2:
  def __init__(self, traj='walking', simrate=60, clock_based=False, state_est=False, dynamics_randomization=False, no_delta=False, reward="iros_paper", history=0):
    self.sim = CassieSim("./cassie/cassiemujoco/cassie.xml")
    self.vis = None

    self.reward_func = reward

    self.clock_based = clock_based
    self.state_est = state_est
    self.no_delta = no_delta
    self.dynamics_randomization = dynamics_randomization

    # Configure reference trajectory to use
    if traj == "aslip":
        self.speeds = np.array([x / 10 for x in range(0, 21)])
        self.trajectories = getAllTrajectories(self.speeds)
        self.num_speeds = len(self.trajectories)
        self.speed = self.speeds[0]
        self.trajectory = self.trajectories[0]
        self.aslip_traj = True
        self.clock_based = False
    else:
        self.aslip_traj = False
        dirname = os.path.dirname(__file__)
        if traj == "walking":
            traj_path = os.path.join(dirname, "trajectory", "stepdata.bin")
        elif traj == "stepping":
            traj_path = os.path.join(dirname, "trajectory", "more-poses-trial.bin")
        self.trajectory = CassieTrajectory(traj_path)
        self.speed = 0

    self.observation_space, self.clock_inds, self.mirrored_obs = self.set_up_state_space()

    # Adds option for state history for FF nets
    self._obs = len(self.observation_space)
    self.history = history

    self.observation_space = np.zeros(self._obs + self._obs * self.history)
    self.action_space = np.zeros(10)

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
    self.phaselen = floor(len(self.trajectory) / self.simrate) - 1 if not self.aslip_traj else self.trajectory.length - 1

    # see include/cassiemujoco.h for meaning of these indices
    self.pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
    self.vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

    self.pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
    self.vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])

    self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])

    # global flat foot orientation, can be useful part of reward function:
    self.global_initial_foot_orient = np.array([-0.24135469773826795, -0.24244324494623198, -0.6659363823866352, 0.6629463911006771])
    self.avg_lfoot_quat = np.zeros(4)
    self.avg_rfoot_quat = np.zeros(4)

    # maybe make ref traj only send relevant idxs?
    ref_pos, ref_vel = self.get_ref_state(self.phase)
    self.prev_action = ref_pos[self.pos_idx]
    self.phase_add = 1

    # Record default dynamics parameters
    self.default_damping = self.sim.get_dof_damping()
    self.default_mass = self.sim.get_body_mass()
    self.default_ipos = self.sim.get_body_ipos()
    self.default_fric = self.sim.get_ground_friction()

    self.critic_state = None

    self.debug = False

  def set_up_state_space(self):

    mjstate_size   = 40
    state_est_size = 46

    speed_size     = 1

    clock_size    = 2
    
    # Find the mirrored trajectory
    if self.aslip_traj:
        ref_traj_size = 18
        mirror_taskspace = [6,7,8,9,10,11,0,1,2,3,4,5,12,13,14,15,16,17]
        mirrored_traj = [x + state_est_size for x in mirror_taskspace] if self.state_est else [x + mjstate_size for x in mirror_taskspace]
    else:
        ref_traj_size = 40
        if self.state_est:
            mirrored_traj = [ 46,  47,  48,  49,  50,  51, -59, -60,  61,  62,  63,  64,  65, -52, -53,  54,  55,  56,  57,  58,  66,  67,  68,  69,  70,  71,-79, -80,  81,  82,  83,  84,  85, -72, -73,  74,  75,  76,  77, 78]
        else:
            mirrored_traj = [ 42,  43,  44,  45,  46,  47, -55, -56,  57,  58,  59,  60,  61, -48, -49,  50,  51,  52,  53,  54,  62,  63,  64,  65,  66,  67, -75, -76,  77,  78,  79,  80,  81, -68, -69,  70,  71,  72,  73, 74]

    # construct mirrored observations for clock-based
    if self.clock_based:
        if self.state_est:
            observation_space = np.zeros(state_est_size + clock_size + speed_size)
            clock_inds = [46, 47]
            mirrored_obs = [0.1, 1, 2, 3, 4, -10, -11, 12, 13, 14, -5, -6, 7, 8, 9, 15, 16, 17, 18, 19, 20, -26, -27, 28, 29, 30, -21, -22, 23, 24, 25, 31, 32, 33, 37, 38, 39, 34, 35, 36, 43, 44, 45, 40, 41, 42, 46, 47, 48]
        else:
            observation_space = np.zeros(mjstate_size + clock_size + speed_size)
            clock_inds = [40, 41]
            mirrored_obs = [0.1, 1, 2, 3, 4, 5, -13, -14, 15, 16, 17, 18, 19, -6, -7, 8, 9, 10, 11, 12, 20, 21, 22, 23, 24, 25, -33, -34, 35, 36, 37, 38, 39, -26, -27, 28, 29, 30, 31, 32, 40, 41, 42]
    
    # construct mirrored observations for traj-based. Here we assume that traj-based has no speed input
    else:
        if self.state_est:
            observation_space = np.zeros(state_est_size + ref_traj_size)
            mirrored_obs = [0.1, 1, 2, 3, 4, -10, -11, 12, 13, 14, -5, -6, 7, 8, 9, 15, 16, 17, 18, 19, 20, -26, -27, 28, 29, 30, -21, -22, 23, 24, 25, 31, 32, 33, 37, 38, 39, 34, 35, 36, 43, 44, 45, 40, 41, 42] + mirrored_traj
        else:
            observation_space = np.zeros(mjstate_size + ref_traj_size)
            mirrored_obs = [0.1, 1, 2, 3, 4, 5, -13, -14, 15, 16, 17, 18, 19, -6, -7, 8, 9, 10, 11, 12, 20, 21, 22, 23, 24, 25, -33, -34, 35, 36, 37, 38, 39, -26, -27, 28, 29, 30, 31, 32] + mirrored_traj
        clock_inds = None

    return observation_space, clock_inds, mirrored_obs

  def step_simulation(self, action):

    if self.aslip_traj and self.phase == self.phaselen - 1:
      ref_pos, ref_vel = self.get_ref_state(0)
    else:
      # maybe make ref traj only send relevant idxs?
      ref_pos, ref_vel = self.get_ref_state(self.phase + self.phase_add)
    
    if not self.no_delta:
      target = action + ref_pos[self.pos_idx]
    else:
      target = action + self.offset
    
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

  def step(self, action, return_omniscient_state=False):
      for _ in range(self.simrate):
          self.step_simulation(action)
          # calculate running average of foot quaternion
          self.avg_lfoot_quat += self.sim.xquat("left-foot")
          self.avg_rfoot_quat += self.sim.xquat("right-foot")
      self.avg_lfoot_quat /= self.simrate
      self.avg_rfoot_quat /= self.simrate
      height = self.sim.qpos()[2]

      self.time  += 1
      self.phase += self.phase_add

      if (self.aslip_traj and self.phase >= self.phaselen) or self.phase > self.phaselen:
          self.phase = 0
          self.counter += 1

      # Early termination
      done = not(height > 0.4 and height < 3.0)

      reward = self.compute_reward(action)

      # reset avg foot quaternion
      self.avg_lfoot_quat = np.zeros(4)
      self.avg_rfoot_quat = np.zeros(4)

      # update previous action
      self.prev_action = action

      # TODO: make 0.3 a variable/more transparent
      if reward < 0.3:
          done = True

      if return_omniscient_state:
        return self.get_full_state(), self.get_omniscient_state(), reward, done, {}
      else:
        return self.get_full_state(), reward, done, {}

  def reset(self):

      self.state_history = [np.zeros(self._obs) for _ in range(self.history+1)]

      if self.aslip_traj:
        random_speed_idx = random.randint(0, self.num_speeds-1)
        self.speed = self.speeds[random_speed_idx]
        # print("current speed: {}".format(self.speed))
        self.trajectory = self.trajectories[random_speed_idx] # switch the current trajectory
        self.phaselen = self.trajectory.length - 1
      else:
        self.speed = (random.randint(0, 10)) / 10
    
      self.phase = random.randint(0, self.phaselen)
      self.time = 0
      self.counter = 0

      qpos, qvel = self.get_ref_state(self.phase)

      self.sim.set_qpos(qpos)
      if self.aslip_traj:
        self.sim.set_qvel(np.zeros(qvel.shape))
      else:
        self.sim.set_qvel(qvel)

      # Randomize dynamics:
      if self.dynamics_randomization:
          damp = self.default_damping
          weak_factor = 0.8
          strong_factor = 1.2
          pelvis_damp_range = [[damp[0], damp[0]], 
                               [damp[1], damp[1]], 
                               [damp[2], damp[2]], 
                               [damp[3], damp[3]], 
                               [damp[4], damp[4]], 
                               [damp[5], damp[5]]]                 # 0->5

          hip_damp_range = [[damp[6]*weak_factor, damp[6]*strong_factor],
                            [damp[7]*weak_factor, damp[7]*strong_factor],
                            [damp[8]*weak_factor, damp[8]*strong_factor]]  # 6->8 and 19->21

          achilles_damp_range = [[damp[9]*weak_factor,  damp[9]*strong_factor],
                                 [damp[10]*weak_factor, damp[10]*strong_factor], 
                                 [damp[11]*weak_factor, damp[11]*strong_factor]] # 9->11 and 22->24

          knee_damp_range     = [[damp[12]*weak_factor, damp[12]*strong_factor]]   # 12 and 25
          shin_damp_range     = [[damp[13]*weak_factor, damp[13]*strong_factor]]   # 13 and 26
          tarsus_damp_range   = [[damp[14], damp[14]]]             # 14 and 27
          heel_damp_range     = [[damp[15], damp[15]]]                           # 15 and 28
          fcrank_damp_range   = [[damp[16]*weak_factor, damp[16]*strong_factor]]   # 16 and 29
          prod_damp_range     = [[damp[17], damp[17]]]                           # 17 and 30
          foot_damp_range     = [[damp[18]*weak_factor, damp[18]*strong_factor]]   # 18 and 31

          side_damp = hip_damp_range + achilles_damp_range + knee_damp_range + shin_damp_range + tarsus_damp_range + heel_damp_range + fcrank_damp_range + prod_damp_range + foot_damp_range
          damp_range = pelvis_damp_range + side_damp + side_damp
          damp_noise = [np.random.uniform(a, b) for a, b in damp_range]

          hi = 1.1
          lo = 0.9
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

          side_mass = hip_mass_range + achilles_mass_range \
                      + knee_mass_range + knee_spring_mass_range \
                      + shin_mass_range + tarsus_mass_range \
                      + heel_spring_mass_range + fcrank_mass_range \
                      + prod_mass_range + foot_mass_range

          mass_range = [[0, 0]] + pelvis_mass_range + side_mass + side_mass
          mass_noise = [np.random.uniform(a, b) for a, b in mass_range]

          delta = 0.0005
          com_noise = [0, 0, 0] + [self.default_ipos[i] + np.random.uniform(-delta, delta) for i in range(3, len(self.default_ipos))]

          fric_noise = [np.random.uniform(0.5, 1.2)] + [np.random.uniform(3e-3, 8e-3)] + list(self.default_fric[2:])

          self.sim.set_dof_damping(np.clip(damp_noise, 0, None))
          self.sim.set_body_mass(np.clip(mass_noise, 0, None))
          self.sim.set_body_ipos(np.clip(com_noise, 0, None))
          self.sim.set_ground_friction(np.clip(fric_noise, 0, None))
      else:
          self.sim.set_dof_damping(self.default_damping)
          self.sim.set_body_mass(self.default_mass)
          self.sim.set_body_ipos(self.default_ipos)
          self.sim.set_ground_friction(self.default_fric)

      self.sim.set_const()

      # Need to reset u? Or better way to reset cassie_state than taking step
      self.cassie_state = self.sim.step_pd(self.u)

      # maybe make ref traj only send relevant idxs?
      ref_pos, ref_vel = self.get_ref_state(self.phase)
      self.prev_action = ref_pos[self.pos_idx]

      actor_state  = self.get_full_state()

      return actor_state

  def reset_for_test(self):

      self.state_history = [np.zeros(self._obs) for _ in range(self.history+1)]

      if self.aslip_traj:
        self.speed = 0
        # print("current speed: {}".format(self.speed))
        self.trajectory = self.trajectories[0] # switch the current trajectory
        self.phaselen = self.trajectory.length - 1
      else:
        self.speed = 0
    
      self.phase = 0
      self.time = 0
      self.counter = 0

      qpos, qvel = self.get_ref_state(self.phase)

    #   self.sim.full_reset()

    #   self.sim.set_qpos(qpos)
    #   if self.aslip_traj:
    #     self.sim.set_qvel(np.zeros(qvel.shape))
    #   else:
    #     self.sim.set_qvel(qvel)

      # Randomize dynamics:
      if self.dynamics_randomization:
          damp = self.default_damping
          weak_factor = 0.8
          strong_factor = 1.2
          pelvis_damp_range = [[damp[0], damp[0]], 
                               [damp[1], damp[1]], 
                               [damp[2], damp[2]], 
                               [damp[3], damp[3]], 
                               [damp[4], damp[4]], 
                               [damp[5], damp[5]]]                 # 0->5

          hip_damp_range = [[damp[6]*weak_factor, damp[6]*strong_factor],
                            [damp[7]*weak_factor, damp[7]*strong_factor],
                            [damp[8]*weak_factor, damp[8]*strong_factor]]  # 6->8 and 19->21

          achilles_damp_range = [[damp[9]*weak_factor,  damp[9]*strong_factor],
                                 [damp[10]*weak_factor, damp[10]*strong_factor], 
                                 [damp[11]*weak_factor, damp[11]*strong_factor]] # 9->11 and 22->24

          knee_damp_range     = [[damp[12]*weak_factor, damp[12]*strong_factor]]   # 12 and 25
          shin_damp_range     = [[damp[13]*weak_factor, damp[13]*strong_factor]]   # 13 and 26
          tarsus_damp_range   = [[damp[14], damp[14]]]             # 14 and 27
          heel_damp_range     = [[damp[15], damp[15]]]                           # 15 and 28
          fcrank_damp_range   = [[damp[16]*weak_factor, damp[16]*strong_factor]]   # 16 and 29
          prod_damp_range     = [[damp[17], damp[17]]]                           # 17 and 30
          foot_damp_range     = [[damp[18]*weak_factor, damp[18]*strong_factor]]   # 18 and 31

          side_damp = hip_damp_range + achilles_damp_range + knee_damp_range + shin_damp_range + tarsus_damp_range + heel_damp_range + fcrank_damp_range + prod_damp_range + foot_damp_range
          damp_range = pelvis_damp_range + side_damp + side_damp
          damp_noise = [np.random.uniform(a, b) for a, b in damp_range]

          hi = 1.1
          lo = 0.9
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

          side_mass = hip_mass_range + achilles_mass_range \
                      + knee_mass_range + knee_spring_mass_range \
                      + shin_mass_range + tarsus_mass_range \
                      + heel_spring_mass_range + fcrank_mass_range \
                      + prod_mass_range + foot_mass_range

          mass_range = [[0, 0]] + pelvis_mass_range + side_mass + side_mass
          mass_noise = [np.random.uniform(a, b) for a, b in mass_range]

          delta = 0.0005
          com_noise = [0, 0, 0] + [self.default_ipos[i] + np.random.uniform(-delta, delta) for i in range(3, len(self.default_ipos))]

          fric_noise = [np.random.uniform(0.5, 1.2)] + [np.random.uniform(3e-3, 8e-3)] + list(self.default_fric[2:])

          self.sim.set_dof_damping(np.clip(damp_noise, 0, None))
          self.sim.set_body_mass(np.clip(mass_noise, 0, None))
          self.sim.set_body_ipos(np.clip(com_noise, 0, None))
          self.sim.set_ground_friction(np.clip(fric_noise, 0, None))
      else:
          self.sim.set_dof_damping(self.default_damping)
          self.sim.set_body_mass(self.default_mass)
          self.sim.set_body_ipos(self.default_ipos)
          self.sim.set_ground_friction(self.default_fric)

      self.sim.set_const()

      # Need to reset u? Or better way to reset cassie_state than taking step
      self.cassie_state = self.sim.step_pd(self.u)

      # maybe make ref traj only send relevant idxs?
      ref_pos, ref_vel = self.get_ref_state(self.phase)
      self.prev_action = ref_pos[self.pos_idx]

      actor_state  = self.get_full_state()

      return actor_state

  # NOTE: this reward is slightly different from the one in Xie et al
  # see notes for details
  def compute_reward(self, action):
      qpos = np.copy(self.sim.qpos())
      qvel = np.copy(self.sim.qvel())

      ref_pos, ref_vel = self.get_ref_state(self.phase)

      if self.reward_func == "jonah_RNN":
          return jonah_RNN_reward(self)
      elif self.reward_func == "aslip":
          return aslip_reward(self, action)
      elif self.reward_func == "aslip_TaskSpace":
          return aslip_TaskSpace_reward(self, action)
      elif self.reward_func == "iros_paper":
          return iros_paper_reward(self)
      else:
          raise NotImplementedError

  # get the corresponding state from the reference trajectory for the current phase
  def get_ref_state(self, phase=None):
      if phase is None:
          phase = self.phase

      if phase > self.phaselen:
          phase = 0

      pos = np.copy(self.trajectory.qpos[phase * self.simrate]) if not self.aslip_traj else np.copy(self.trajectory.qpos[phase])

      # this is just setting the x to where it "should" be given the number
      # of cycles
      # pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * self.counter
      
      # ^ should only matter for COM error calculation,
      # gets dropped out of state variable for input reasons

      ###### Setting variable speed  #########
      pos[0] *= self.speed
      pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * self.counter * self.speed
      ######                          ########

      # setting lateral distance target to 0?
      # regardless of reference trajectory?
      pos[1] = 0

      vel = np.copy(self.trajectory.qvel[phase * self.simrate]) if not self.aslip_traj else np.copy(self.trajectory.qvel[phase])
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

      if self.clock_based and not self.aslip_traj:
        clock = [np.sin(2 * np.pi *  self.phase / self.phaselen),
                 np.cos(2 * np.pi *  self.phase / self.phaselen)]
        
        ext_state = np.concatenate((clock, [self.speed]))

      elif self.aslip_traj:
        if(self.phase == 0):
            ext_state = np.concatenate(get_ref_aslip_ext_state(self, self.phaselen - 1))
        else:
            ext_state = np.concatenate(get_ref_aslip_ext_state(self, self.phase))
      else:
        ext_state = np.concatenate([ref_pos[self.pos_index], ref_vel[self.vel_index]])

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
          state = np.concatenate([robot_state, ext_state])
      else:
          state = np.concatenate([qpos[self.pos_index], qvel[self.vel_index], ext_state])

      self.state_history.insert(0, state)
      self.state_history = self.state_history[:self.history+1]

      return np.concatenate(self.state_history)

  """ Currently unused, commenting out for now.
  def get_omniscient_state(self):
      full_state = self.get_full_state()
      omniscient_state = np.hstack((full_state, self.sim.get_dof_damping(), self.sim.get_body_mass(), self.sim.get_body_ipos(), self.sim.get_ground_friction))
      return omniscient_state
  """

  def render(self):
      if self.vis is None:
          self.vis = CassieVis(self.sim, "./cassie/cassiemujoco/cassie.xml")

      return self.vis.draw(self.sim)

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


# qpos layout
# [ 0] Pelvis x
# [ 1] Pelvis y
# [ 2] Pelvis z
# [ 3] Pelvis orientation qw
# [ 4] Pelvis orientation qx
# [ 5] Pelvis orientation qy
# [ 6] Pelvis orientation qz
# [ 7] Left hip roll         (Motor [0])
# [ 8] Left hip yaw          (Motor [1])
# [ 9] Left hip pitch        (Motor [2])
# [10] Left achilles rod qw
# [11] Left achilles rod qx
# [12] Left achilles rod qy
# [13] Left achilles rod qz
# [14] Left knee             (Motor [3])
# [15] Left shin                        (Joint [0])
# [16] Left tarsus                      (Joint [1])
# [17] Left heel spring
# [18] Left foot crank
# [19] Left plantar rod
# [20] Left foot             (Motor [4], Joint [2])
# [21] Right hip roll        (Motor [5])
# [22] Right hip yaw         (Motor [6])
# [23] Right hip pitch       (Motor [7])
# [24] Right achilles rod qw
# [25] Right achilles rod qx
# [26] Right achilles rod qy
# [27] Right achilles rod qz
# [28] Right knee            (Motor [8])
# [29] Right shin                       (Joint [3])
# [30] Right tarsus                     (Joint [4])
# [31] Right heel spring
# [32] Right foot crank
# [33] Right plantar rod
# [34] Right foot            (Motor [9], Joint [5])

# qvel layout
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
