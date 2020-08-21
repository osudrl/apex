import numpy as np
import pickle
from cassie.trajectory.aslip_trajectory import get_ref_aslip_ext_state, get_ref_aslip_unaltered_state, get_ref_aslip_global_state

def aslip_old_reward(self, action):

    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    ref_pos, ref_vel = self.get_ref_state(self.phase)

    # TODO: should be variable; where do these come from?
    # TODO: see magnitude of state variables to gauge contribution to reward
    weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]

    # state info
    com_pos = qpos[0:3]
    lfoot_pos = self.cassie_state.leftFoot.position[:]
    rfoot_pos = self.cassie_state.rightFoot.position[:]
    com_vel = self.cassie_state.pelvis.translationalVelocity
    
    footpos_error     = 0
    com_vel_error     = 0
    action_penalty    = 0
    foot_orient_penalty = 0
    straight_diff = 0

    phase_to_match = self.phase

    ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_unaltered_state(self, phase_to_match)

    for j in [0, 1, 2]:
        footpos_error += np.linalg.norm(lfoot_pos[j] - ref_lfoot[j]) +  np.linalg.norm(rfoot_pos[j] - ref_rfoot[j])

    for j in [0, 1, 2]:
        com_vel_error += np.linalg.norm(com_vel[j] - ref_cvel[j])

    # action penalty
    action_penalty = np.linalg.norm(action - self.prev_action)

    # foot orientation penalty
    foot_orient_penalty = self.l_foot_orient_cost + self.r_foot_orient_cost

    # straight difference penalty
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    
    reward = 0.3 * np.exp(-footpos_error) +    \
                0.3 * np.exp(-com_vel_error) +    \
                0.1 * np.exp(-action_penalty) +     \
                0.2 * np.exp(-foot_orient_penalty) + \
                0.1 * np.exp(-straight_diff)

    if self.debug:
        print("reward: {10}\nfoot:\t{0:.2f}, % = {1:.2f}\ncom_vel:\t{2:.2f}, % = {3:.2f}\naction_penalty:\t{4:.2f}, % = {5:.2f}\nfoot_orient_penalty:\t{6:.2f}, % = {7:.2f}\nstraight_diff:\t{8:.2f}, % = {9:.2f}\n\n".format(
        0.3 * np.exp(-footpos_error),          0.3 * np.exp(-footpos_error) / reward * 100,
        0.3 * np.exp(-com_vel_error),          0.3 * np.exp(-com_vel_error) / reward * 100,
        0.1 * np.exp(-action_penalty),         0.1 * np.exp(-action_penalty) / reward * 100,
        0.2 * np.exp(-foot_orient_penalty),    0.2 * np.exp(-foot_orient_penalty) / reward * 100,
        0.1  * np.exp(-straight_diff),         0.1  * np.exp(-straight_diff) / reward * 100,
        reward
        )
        )
        print("actual speed: {}\tdesired_speed: {}\tcommanded speed: {}".format(np.linalg.norm(qvel[0:3]), np.linalg.norm(ref_cvel), self.speed))
    return reward

"""

def aslip_oldMujoco_reward(self, action):

    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    ref_pos, ref_vel = self.get_ref_state(self.phase)

    # TODO: should be variable; where do these come from?
    # TODO: see magnitude of state variables to gauge contribution to reward
    # weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]

    weight = [0.05, 0.05, 0.25, 0.25, 0.05, 0.05, 0.05, 0.25, 0.25, 0.05]
    reward_pos_idx = [3,4,5,6,7, 8, 9, 14, 20, 21, 22, 23, 28, 34]

    #weight = [.1] * 10

    # mujoco state info
    joint_error = 0
    com_pos = qpos[0:3]
    # lfoot_pos = self.l_foot_pos - qpos[0:3]
    # rfoot_pos = self.r_foot_pos - qpos[0:3]
    lfoot_pos = self.l_foot_pos
    rfoot_pos = self.r_foot_pos
    com_vel = qvel[0:3]

    footpos_error     = 0
    com_vel_error     = 0
    action_penalty    = 0
    foot_orient_penalty = 0
    straight_diff = 0

    phase_to_match = self.phase

    # ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_ext_state(self, self.cassie_state, self.last_pelvis_pos, phase_to_match, offset=self.vertOffset)
    # ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_unaltered_state(self, phase_to_match)
    ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_global_state(self, phase_to_match)

    for j in [0, 1, 2]:
        footpos_error += np.linalg.norm(lfoot_pos[j] - ref_lfoot[j]) +  np.linalg.norm(rfoot_pos[j] - ref_rfoot[j])

    # # enforce distance between feet and com
    # footpos_error += np.linalg.norm(ref_lfoot - lfoot_pos) + np.linalg.norm(ref_rfoot - rfoot_pos)

    # com_vel_error += np.linalg.norm(com_vel - ref_cvel)

    for j in [0, 1, 2]:
        com_vel_error += np.linalg.norm(com_vel[j] - ref_cvel[j])

    # each joint pos, skipping feet
    for i, j in enumerate(reward_pos_idx):
        target = ref_pos[j]
        actual = qpos[j]

        if j == 20 or j == 34:
            joint_error += 0
        else:
            joint_error += (target - actual) ** 2

    # action penalty
    action_penalty = np.linalg.norm(action - self.prev_action)

    # foot orientation penalty
    foot_orient_penalty = self.l_foot_orient_cost + self.r_foot_orient_cost

    # straight difference penalty
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    
    
    reward = 0.3 * np.exp(-footpos_error) +    \
                0.3 * np.exp(-com_vel_error) +    \
                0.1 * np.exp(-action_penalty) +     \
                0.2 * np.exp(-foot_orient_penalty) + \
                0.1 * np.exp(-straight_diff)

    if self.debug:
        print("reward: {10}\nfoot:\t{0:.2f}, % = {1:.2f}\ncom_vel:\t{2:.2f}, % = {3:.2f}\naction_penalty:\t{4:.2f}, % = {5:.2f}\nfoot_orient_penalty:\t{6:.2f}, % = {7:.2f}\nstraight_diff:\t{8:.2f}, % = {9:.2f}\n\n".format(
        0.3 * np.exp(-footpos_error),          0.3 * np.exp(-footpos_error) / reward * 100,
        0.3 * np.exp(-com_vel_error),          0.3 * np.exp(-com_vel_error) / reward * 100,
        0.1 * np.exp(-action_penalty),         0.1 * np.exp(-action_penalty) / reward * 100,
        0.2 * np.exp(-foot_orient_penalty),    0.2 * np.exp(-foot_orient_penalty) / reward * 100,
        0.1  * np.exp(-straight_diff),         0.1  * np.exp(-straight_diff) / reward * 100,
        reward
        )
        )
        print("actual speed: {}\tdesired_speed: {}\tcommanded speed: {}".format(np.linalg.norm(qvel[0:3]), np.linalg.norm(ref_cvel), self.speed))
    return reward


def aslip_joint_reward(self, action):

    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    ref_pos, ref_vel = self.get_ref_state(self.phase)

    # TODO: should be variable; where do these come from?
    # TODO: see magnitude of state variables to gauge contribution to reward
    reward_pos_idx = [3,4,5,6,7, 8, 9, 14, 20, 21, 22, 23, 28, 34]

    # mujoco state info
    com_pos = qpos[0:3]
    # lfoot_pos = self.l_foot_pos - qpos[0:3]
    # rfoot_pos = self.r_foot_pos - qpos[0:3]
    lfoot_pos = self.l_foot_pos
    rfoot_pos = self.r_foot_pos
    com_vel = qvel[0:3]
    
    joint_error = 0
    footpos_error     = 0
    com_vel_error     = 0
    # action_penalty    = 0
    # com_orient_error  = 0
    straightheight_diff = 0

    phase_to_match = self.phase

    # ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_ext_state(self, self.cassie_state, self.last_pelvis_pos, phase_to_match, offset=self.vertOffset)
    # ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_unaltered_state(self, phase_to_match)
    ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_global_state(self, phase_to_match)

    for j in [0, 1, 2]:
        footpos_error += np.linalg.norm(lfoot_pos[j] - ref_lfoot[j]) +  np.linalg.norm(rfoot_pos[j] - ref_rfoot[j])

    # # enforce distance between feet and com
    # footpos_error += 10 * (np.linalg.norm(ref_lfoot - lfoot_pos) + np.linalg.norm(ref_rfoot - rfoot_pos))

    for j in [0, 1, 2]:
        com_vel_error += np.linalg.norm(com_vel[j] - ref_cvel[j])

    # com_vel_error += 10 * (np.abs(self.speed - com_vel[0]))

    # each joint pos, skipping feet
    for i, j in enumerate(reward_pos_idx):
        target = ref_pos[j]
        actual = qpos[j]

        if j == 20 or j == 34:
            joint_error += 0
        else:
            joint_error += (target - actual) ** 2

    # # action penalty
    # action_penalty = np.linalg.norm(action - self.prev_action)

    # # com orientation penalty
    # com_orient_error = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))

    # straight difference penalty
    straight_diff = np.abs(qpos[1])
    height_diff = np.abs(qpos[2] - ref_cpos[2])
    if straight_diff < 0.05: # allow some side to side
        straight_diff = 0
    straightheight_diff = 10 * (straight_diff + 5 * height_diff)
    straightheight_diff = 10 * (np.linalg.norm(ref_cpos[1:3] - com_pos[1:3])) # only match y and z, don't match x
    
    reward = 0.4 * np.exp(-joint_error) +    \
                0.2 * np.exp(-footpos_error) +    \
                0.2 * np.exp(-com_vel_error) +    \
                0.2 * np.exp(-straightheight_diff)

    if self.debug:
        print("reward: {8}\nfoot:\t{0:.2f}, % = {1:.2f}\ncom_vel:\t{2:.2f}, % = {3:.2f}\njoint:\t{4:.2f}, % = {5:.2f}\nstraightheight_diff:\t{6:.2f}, % = {7:.2f}\n\n".format(
        0.2 * np.exp(-footpos_error),          0.2 * np.exp(-footpos_error) / reward * 100,
        0.2 * np.exp(-com_vel_error),          0.2 * np.exp(-com_vel_error) / reward * 100,
        0.4 * np.exp(-joint_error),            0.4 * np.exp(-joint_error) / reward * 100,
        0.2  * np.exp(-straightheight_diff),         0.2  * np.exp(-straightheight_diff) / reward * 100,
        reward
        )
        )
        print("actual speed: {}\tdesired_speed: {}\tcommanded speed: {}".format(np.linalg.norm(qvel[0:3]), np.linalg.norm(ref_cvel), self.speed))
    return reward


def aslip_comorientheight_reward(self, action):

    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    ref_pos, ref_vel = self.get_ref_state(self.phase)

    # TODO: should be variable; where do these come from?
    # TODO: see magnitude of state variables to gauge contribution to reward
    # weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]

    weight = [0.05, 0.05, 0.25, 0.25, 0.05, 0.05, 0.05, 0.25, 0.25, 0.05]
    reward_pos_idx = [3,4,5,6,7, 8, 9, 14, 20, 21, 22, 23, 28, 34]

    #weight = [.1] * 10

    # mujoco state info
    joint_error = 0
    com_pos = qpos[0:3]
    # lfoot_pos = self.l_foot_pos - qpos[0:3]
    # rfoot_pos = self.r_foot_pos - qpos[0:3]
    lfoot_pos = self.l_foot_pos
    rfoot_pos = self.r_foot_pos
    com_vel = qvel[0:3]

    footpos_error     = 0
    com_vel_error     = 0
    action_penalty    = 0
    com_orient_error  = 0
    straightheight_diff = 0

    phase_to_match = self.phase

    # ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_ext_state(self, self.cassie_state, self.last_pelvis_pos, phase_to_match, offset=self.vertOffset)
    # ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_unaltered_state(self, phase_to_match)
    ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_global_state(self, phase_to_match)

    for j in [0, 1, 2]:
        footpos_error += np.linalg.norm(lfoot_pos[j] - ref_lfoot[j]) +  np.linalg.norm(rfoot_pos[j] - ref_rfoot[j])

    # # enforce distance between feet and com
    # footpos_error += np.linalg.norm(ref_lfoot - lfoot_pos) + np.linalg.norm(ref_rfoot - rfoot_pos)

    for j in [0, 1, 2]:
        com_vel_error += np.linalg.norm(com_vel[j] - ref_cvel[j])

    # com_vel_error += np.linalg.norm(com_vel - ref_cvel)

    # each joint pos, skipping feet
    for i, j in enumerate(reward_pos_idx):
        target = ref_pos[j]
        actual = qpos[j]

        if j == 20 or j == 34:
            joint_error += 0
        else:
            joint_error += (target - actual) ** 2

    # action penalty
    action_penalty = np.linalg.norm(action - self.prev_action)

    # com orientation penalty
    com_orient_error = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))

    # straight difference penalty
    straight_diff = np.abs(qpos[1])
    height_diff = np.abs(qpos[2] - ref_cpos[2])
    if straight_diff < 0.05: # allow some side to side
        straight_diff = 0
    straightheight_diff = straight_diff + 5 * height_diff
    
    reward = 0.3 * np.exp(-footpos_error) +    \
                0.3 * np.exp(-com_vel_error) +    \
                0.1 * np.exp(-action_penalty) +     \
                0.1 * np.exp(-com_orient_error) + \
                0.2 * np.exp(-straightheight_diff)

    if self.debug:
        print("reward: {10}\nfoot:\t{0:.2f}, % = {1:.2f}\ncom_vel:\t{2:.2f}, % = {3:.2f}\naction_penalty:\t{4:.2f}, % = {5:.2f}\ncom_orient_error:\t{6:.2f}, % = {7:.2f}\nstraightheight_diff:\t{8:.2f}, % = {9:.2f}\n\n".format(
        0.3 * np.exp(-footpos_error),          0.3 * np.exp(-footpos_error) / reward * 100,
        0.3 * np.exp(-com_vel_error),          0.3 * np.exp(-com_vel_error) / reward * 100,
        0.1 * np.exp(-action_penalty),         0.1 * np.exp(-action_penalty) / reward * 100,
        0.1 * np.exp(-com_orient_error),       0.1 * np.exp(-com_orient_error) / reward * 100,
        0.2  * np.exp(-straightheight_diff),         0.2  * np.exp(-straightheight_diff) / reward * 100,
        reward
        )
        )
        print("actual speed: {}\tdesired_speed: {}\tcommanded speed: {}".format(np.linalg.norm(qvel[0:3]), np.linalg.norm(ref_cvel), self.speed))
    return reward


# USING Mujoco State
def aslip_TaskSpaceMujoco_reward(self, action):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    
    phase_to_match = self.phase

    # offset now directly in trajectories
    # ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_ext_state(self, self.cassie_state, self.last_pelvis_pos, phase_to_match, offset=self.vertOffset)
    # ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_unaltered_state(self, phase_to_match)
    ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_global_state(self, phase_to_match)

    # mujoco state info
    com_pos = qpos[0:3]
    # lfoot_pos = self.l_foot_pos - qpos[0:3]
    # rfoot_pos = self.r_foot_pos - qpos[0:3]
    lfoot_pos = self.l_foot_pos
    rfoot_pos = self.r_foot_pos
    com_vel = qvel[0:3]

    footpos_error        = 0
    compos_error         = 0
    comvel_error         = 0
    # action_penalty       = 0

    # enforce distance between feet and com
    footpos_error += np.linalg.norm(ref_lfoot - lfoot_pos) + np.linalg.norm(ref_rfoot - lfoot_pos)

    # enforce com position matching
    compos_error += np.linalg.norm(ref_cpos[1:3] - com_pos[1:3]) # only match y and z, don't match x

    # try to match com velocity
    comvel_error += np.linalg.norm(ref_cvel - com_vel)

    # action smoothing penalty term
    # action_penalty = np.linalg.norm(action - self.prev_action)

    if self.debug:
        print("ref_rfoot: {}  rfoot: {}".format(ref_rfoot, rfoot_pos))
        print("ref_lfoot: {}  lfoot: {}".format(ref_lfoot, lfoot_pos))
        print(footpos_error)
        print("ref_cpos:  {}   cpos: {}".format(ref_cpos, com_pos))
        print(compos_error)

    reward = (10/30) * np.exp(-footpos_error) +    \
             (10/30) * np.exp(-compos_error) +    \
             (10/30) * np.exp(-comvel_error)
            #  (6/30) * np.exp(-action_penalty)
    
    # like a height termination 
    # if com_pos[2] < 0.7:
    #     reward = 0

    if self.debug:
        print("reward: {6}\nfoot_pos:\t{0:.2f}, % = {1:.2f}\ncom_pos:\t{2:.2f}, % = {3:.2f}\ncomvel_error:\t{4:.2f}, % = {5:.2f}\n\n".format(
        (10/30) * np.exp(-footpos_error),          (10/30) * np.exp(-footpos_error) / reward * 100,
        (10/30) * np.exp(-compos_error),           (10/30) * np.exp(-compos_error) / reward * 100,
        (10/30) * np.exp(-comvel_error),           (10/30) * np.exp(-comvel_error) / reward * 100,
        # (6/30) * np.exp(-action_penalty),         (6/30) * np.exp(-action_penalty) / reward * 100,
        reward
        )
        )
        print("actual speed: {}\tdesired_speed: {}\tcommanded speed: {}".format(np.linalg.norm(qvel[0:3]), np.linalg.norm(ref_cvel), self.speed))
    return reward


# Using Mujoco State
def aslip_DirectMatchMujoco_reward(self, action):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    
    phase_to_match = self.phase

    # offset now directly in trajectories
    # ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_ext_state(self, self.cassie_state, self.last_pelvis_pos, phase_to_match, offset=self.vertOffset)
    ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_unaltered_state(self, phase_to_match)

    # mujoco state info
    com_pos = qpos[0:3]
    lfoot_pos = self.l_foot_pos - qpos[0:3]
    rfoot_pos = self.r_foot_pos - qpos[0:3]
    com_vel = qvel[0:3]
    lfoot_vel = self.l_foot_vel
    rfoot_vel = self.r_foot_vel

    footpos_error        = 0
    footvel_error        = 0
    compos_error         = 0
    comvel_error         = 0

    # enforce distance between feet and com
    footpos_error += 10 * (np.linalg.norm(ref_lfoot - lfoot_pos) + np.linalg.norm(ref_rfoot - lfoot_pos))
    
    # enforce velocity of
    footvel_error += 10 * (np.linalg.norm(ref_lvel - lfoot_vel) + np.linalg.norm(ref_rvel - rfoot_vel))

    # enforce com position matching
    compos_error += 10 * (np.linalg.norm(ref_cpos[1:3] - com_pos[1:3])) # only match y and z, don't match x

    # try to match com velocity
    comvel_error += 10 * (np.linalg.norm(ref_cvel - com_vel))

    if self.debug:
        print("ref_rfoot: {}  rfoot: {}".format(ref_rfoot, rfoot_pos))
        print("ref_lfoot: {}  lfoot: {}".format(ref_lfoot, lfoot_pos))
        print(footpos_error)
        print("ref_cpos:  {}   cpos: {}".format(ref_cpos, com_pos))
        print(compos_error)

    reward = 0.25 * np.exp(-footpos_error) +    \
             0.25 * np.exp(-footvel_error) +    \
             0.25 * np.exp(-compos_error) +    \
             0.25 * np.exp(-comvel_error)
    
    # like a height termination 
    # if com_pos[2] < 0.7:
    #     reward = 0

    if self.debug:
        print("reward: {8}\nfoot_pos:\t{0:.2f}, % = {1:.2f}\nfoot_vel:\t{2:.2f}, % = {3:.2f}\ncom_pos:\t{4:.2f}, % = {5:.2f}\ncom_vel:\t{6:.2f}, % = {7:.2f}\n\n".format(
        0.2 * np.exp(-footpos_error),          0.2 * np.exp(-footpos_error) / reward * 100,
        0.2 * np.exp(-footvel_error),           0.2 * np.exp(-footvel_error) / reward * 100,
        0.2 * np.exp(-compos_error),          0.2 * np.exp(-compos_error) / reward * 100,
        0.2 * np.exp(-comvel_error),    0.2 * np.exp(-comvel_error) / reward * 100,
        reward
        )
        )
        print("actual speed: {}\tdesired_speed: {}\tcommanded speed: {}".format(np.linalg.norm(qvel[0:3]), np.linalg.norm(ref_cvel), self.speed))
    return reward


# USING State Est
def aslip_TaskSpaceStateEst_reward(self, action):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    
    phase_to_match = self.phase

    # offset now directly in trajectories
    # ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_ext_state(self, self.cassie_state, self.last_pelvis_pos, phase_to_match, offset=self.vertOffset)
    ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_unaltered_state(self, phase_to_match)

    # state estimate info
    com_pos = self.cassie_state.pelvis.position[:]
    com_pos[2] - self.cassie_state.terrain.height
    lfoot_pos = self.cassie_state.leftFoot.position[:]
    rfoot_pos = self.cassie_state.rightFoot.position[:]
    # lfoot_pos = [com_pos[i] + lfoot_pos[i] for i in range(3)]
    # rfoot_pos = [com_pos[i] + rfoot_pos[i] for i in range(3)]
    com_vel = self.cassie_state.pelvis.translationalVelocity[:]

    footpos_error        = 0
    compos_error         = 0
    straight_diff        = 0
    comvel_error         = 0
    # action_penalty       = 0

    # enforce distance between feet and com
    footpos_error += np.linalg.norm(ref_lfoot - lfoot_pos) + np.linalg.norm(ref_rfoot - lfoot_pos)

    # enforce com position matching
    compos_error += np.linalg.norm(ref_cpos[1:3] - com_pos[1:3]) # only match y and z, don't match x

    # try to match com velocity
    comvel_error += np.linalg.norm(ref_cvel - com_vel)

    # action smoothing penalty term
    # action_penalty = np.linalg.norm(action - self.prev_action)

    if self.debug:
        print("ref_rfoot: {}  rfoot: {}".format(ref_rfoot, rfoot_pos))
        print("ref_lfoot: {}  lfoot: {}".format(ref_lfoot, lfoot_pos))
        print(footpos_error)
        print("ref_cpos:  {}   cpos: {}".format(ref_cpos, com_pos))
        print(compos_error)

    reward = (10/30) * np.exp(-footpos_error) +    \
             (10/30) * np.exp(-compos_error) +    \
             (10/30) * np.exp(-comvel_error)
            #  (6/30) * np.exp(-action_penalty)
    
    # like a height termination 
    # if com_pos[2] < 0.7:
    #     reward = 0

    if self.debug:
        print("reward: {6}\nfoot_pos:\t{0:.2f}, % = {1:.2f}\ncom_pos:\t{2:.2f}, % = {3:.2f}\ncom_vel:\t{4:.2f}, % = {5:.2f}\n\n".format(
        (10/30) * np.exp(-footpos_error),          (10/30) * np.exp(-footpos_error) / reward * 100,
        (10/30) * np.exp(-compos_error),           (10/30) * np.exp(-compos_error) / reward * 100,
        (10/30) * np.exp(-comvel_error),           (10/30) * np.exp(-comvel_error) / reward * 100,
        # (6/30) * np.exp(-action_penalty),         (6/30) * np.exp(-action_penalty) / reward * 100,
        reward
        )
        )
        print("actual speed: {}\tdesired_speed: {}".format(np.linalg.norm(qvel[0:3]), np.linalg.norm(ref_cvel)))
    return reward

# USING State Est
def aslip_DirectMatchStateEst_reward(self, action):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    
    phase_to_match = self.phase

    # offset now directly in trajectories
    # ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_ext_state(self, self.cassie_state, self.last_pelvis_pos, phase_to_match, offset=self.vertOffset)
    ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_unaltered_state(self, phase_to_match)

    # state estimate info
    com_pos = self.cassie_state.pelvis.position[:]
    com_pos[2] - self.cassie_state.terrain.height
    lfoot_pos = self.cassie_state.leftFoot.position[:]
    rfoot_pos = self.cassie_state.rightFoot.position[:]
    lfoot_pos = [com_pos[i] + lfoot_pos[i] for i in range(3)]
    rfoot_pos = [com_pos[i] + rfoot_pos[i] for i in range(3)]
    com_vel = self.cassie_state.pelvis.translationalVelocity[:]
    lfoot_vel = self.cassie_state.leftFoot.footTranslationalVelocity[:]
    rfoot_vel = self.cassie_state.rightFoot.footTranslationalVelocity[:]

    footpos_error        = 0
    footvel_error        = 0
    compos_error         = 0
    comvel_error         = 0

    # enforce distance between feet and com
    footpos_error += 10 * (np.linalg.norm(ref_lfoot - lfoot_pos) + np.linalg.norm(ref_rfoot - lfoot_pos))
    
    # enforce velocity of
    footvel_error += 10 * (np.linalg.norm(ref_lvel - lfoot_vel) + np.linalg.norm(ref_rvel - rfoot_vel))

    # enforce com position matching
    compos_error += 10 * (np.linalg.norm(ref_cpos[1:3] - com_pos[1:3])) # only match y and z, don't match x

    # try to match com velocity
    comvel_error += 10 * (np.linalg.norm(ref_cvel - com_vel))

    if self.debug:
        print("ref_rfoot: {}  rfoot: {}".format(ref_rfoot, rfoot_pos))
        print("ref_lfoot: {}  lfoot: {}".format(ref_lfoot, lfoot_pos))
        print(footpos_error)
        print("ref_cpos:  {}   cpos: {}".format(ref_cpos, com_pos))
        print(compos_error)

    reward = 0.25 * np.exp(-footpos_error) +    \
             0.25 * np.exp(-footvel_error) +    \
             0.25 * np.exp(-compos_error) +    \
             0.25 * np.exp(-comvel_error)
    
    # like a height termination 
    # if com_pos[2] < 0.7:
    #     reward = 0

    if self.debug:
        print("reward: {8}\nfoot_pos:\t{0:.2f}, % = {1:.2f}\nfoot_vel:\t{2:.2f}, % = {3:.2f}\ncom_pos:\t{4:.2f}, % = {5:.2f}\ncom_vel:\t{6:.2f}, % = {7:.2f}\n\n".format(
        0.2 * np.exp(-footpos_error),          0.2 * np.exp(-footpos_error) / reward * 100,
        0.2 * np.exp(-footvel_error),           0.2 * np.exp(-footvel_error) / reward * 100,
        0.2 * np.exp(-compos_error),          0.2 * np.exp(-compos_error) / reward * 100,
        0.2 * np.exp(-comvel_error),    0.2 * np.exp(-comvel_error) / reward * 100,
        reward
        )
        )
        print("actual speed: {}\tdesired_speed: {}".format(np.linalg.norm(qvel[0:3]), np.linalg.norm(ref_cvel)))
    return reward

### Reward from rss submission:

# def aslip_TaskSpace_reward(self, action):
#     qpos = np.copy(self.sim.qpos())
#     qvel = np.copy(self.sim.qvel())

#     ref_pos, ref_vel = self.get_ref_state(self.phase)

#     footpos_error     = 0
#     com_vel_error     = 0
#     action_penalty    = 0
#     foot_orient_penalty = 0
#     straight_diff = 0

#     # enforce distance between feet and com
#     ref_rfoot, ref_lfoot  = get_ref_footdist(self, self.phase + 1)

#     # left foot
#     lfoot = self.cassie_state.leftFoot.position[:]
#     rfoot = self.cassie_state.rightFoot.position[:]
#     for j in [0, 1, 2]:
#         footpos_error += np.linalg.norm(lfoot[j] - ref_lfoot[j]) +  np.linalg.norm(rfoot[j] - ref_rfoot[j])
    
#     if self.debug:
#         print("ref_rfoot: {}  rfoot: {}".format(ref_rfoot, rfoot))
#         print("ref_lfoot: {}  lfoot: {}".format(ref_lfoot, lfoot))
#         print(footpos_error)

#     # try to match com velocity
#     ref_cvel = get_ref_com_vel(self, self.phase + 1)

#     # center of mass vel: x, y, z
#     cvel = self.cassie_state.pelvis.translationalVelocity
#     for j in [0, 1, 2]:
#         com_vel_error += np.linalg.norm(cvel[j] - ref_cvel[j])

#     # # each joint pos, skipping feet
#     # for i, j in enumerate(self.reward_pos_idx):
#     #     target = ref_pos[j]
#     #     actual = qpos[j]

#     #     if j == 20 or j == 34:
#     #         joint_error += 0
#     #     else:
#     #         joint_error += (target - actual) ** 2

#     # action penalty
#     action_penalty = np.linalg.norm(action - self.prev_action)

#     # foot orientation penalty
#     foot_orient_penalty = np.linalg.norm(self.avg_rfoot_quat - self.global_initial_foot_orient) + np.linalg.norm(self.avg_lfoot_quat - self.global_initial_foot_orient)

#     # straight difference penalty
#     straight_diff = np.abs(qpos[1])
#     if straight_diff < 0.05:
#         straight_diff = 0

#     reward = 0.3 * np.exp(-footpos_error) +    \
#                 0.3 * np.exp(-com_vel_error) +    \
#                 0.1 * np.exp(-action_penalty) +     \
#                 0.2 * np.exp(-foot_orient_penalty) + \
#                 0.1 * np.exp(-straight_diff)

#     if self.debug:
#         print("reward: {10}\nfoot:\t{0:.2f}, % = {1:.2f}\ncom_vel:\t{2:.2f}, % = {3:.2f}\naction_penalty:\t{4:.2f}, % = {5:.2f}\nfoot_orient_penalty:\t{6:.2f}, % = {7:.2f}\nstraight_diff:\t{8:.2f}, % = {9:.2f}\n\n".format(
#         0.3 * np.exp(-footpos_error),          0.3 * np.exp(-footpos_error) / reward * 100,
#         0.3 * np.exp(-com_vel_error),          0.3 * np.exp(-com_vel_error) / reward * 100,
#         0.1 * np.exp(-action_penalty),         0.1 * np.exp(-action_penalty) / reward * 100,
#         0.2 * np.exp(-foot_orient_penalty),    0.2 * np.exp(-foot_orient_penalty) / reward * 100,
#         0.1  * np.exp(-straight_diff),         0.1  * np.exp(-straight_diff) / reward * 100,
#         reward
#         )
#         )
#         print("actual speed: {}\tdesired_speed: {}".format(qvel[0], self.speed))
#     return reward


Old rewards

def aslip_strict_reward(self, action):

    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    ref_pos, ref_vel = self.get_ref_state(self.phase)

    # TODO: should be variable; where do these come from?
    # TODO: see magnitude of state variables to gauge contribution to reward
    # weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]

    weight = [0.05, 0.05, 0.25, 0.25, 0.05, 0.05, 0.05, 0.25, 0.25, 0.05]

    #weight = [.1] * 10

    # mujoco state info
    joint_error = 0
    com_pos = qpos[0:3]
    # lfoot_pos = self.l_foot_pos - qpos[0:3]
    # rfoot_pos = self.r_foot_pos - qpos[0:3]
    lfoot_pos = self.l_foot_pos
    rfoot_pos = self.r_foot_pos
    com_vel = qvel[0:3]

    footpos_error     = 0
    com_vel_error     = 0
    # action_penalty    = 0
    foot_orient_penalty = 0
    straight_diff = 0

    phase_to_match = self.phase + 1

    # ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_ext_state(self, self.cassie_state, self.last_pelvis_pos, phase_to_match, offset=self.vertOffset)
    # ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_unaltered_state(self, phase_to_match)
    ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_global_state(self, phase_to_match)

    # enforce distance between feet and com
    footpos_error += 10 * np.linalg.norm(ref_lfoot - lfoot_pos) + np.linalg.norm(ref_rfoot - rfoot_pos)

    com_vel_error += 10 * np.linalg.norm(com_vel - ref_cvel)

    # each joint pos, skipping feet
    for i, j in enumerate(self.pos_idx):
        target = ref_pos[j]
        actual = qpos[j]

        if j == 20 or j == 34:
            joint_error += 0
        else:
            joint_error += 10 * (target - actual) ** 2

    # action penalty
    # action_penalty = np.linalg.norm(action - self.prev_action)

    # foot orientation penalty
    foot_orient_penalty = self.l_foot_orient_cost + self.r_foot_orient_cost

    # straight difference penalty
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    
    
    reward = 0.4 * np.exp(-footpos_error) +    \
                0.3 * np.exp(-com_vel_error) +    \
                0.2 * np.exp(-foot_orient_penalty) + \
                0.1 * np.exp(-straight_diff)

    if self.debug:
        print("reward: {8}\nfoot:\t{0:.2f}, % = {1:.2f}\ncom_vel:\t{2:.2f}, % = {3:.2f}\nfoot_orient_penalty:\t{4:.2f}, % = {5:.2f}\nstraight_diff:\t{6:.2f}, % = {7:.2f}\n\n".format(
        0.3 * np.exp(-footpos_error),          0.3 * np.exp(-footpos_error) / reward * 100,
        0.3 * np.exp(-com_vel_error),          0.3 * np.exp(-com_vel_error) / reward * 100,
        # 0.1 * np.exp(-action_penalty),         0.1 * np.exp(-action_penalty) / reward * 100,
        0.2 * np.exp(-foot_orient_penalty),    0.2 * np.exp(-foot_orient_penalty) / reward * 100,
        0.1  * np.exp(-straight_diff),         0.1  * np.exp(-straight_diff) / reward * 100,
        reward
        )
        )
        print("actual speed: {}\tdesired_speed: {}".format(np.linalg.norm(qvel[0:3]), np.linalg.norm(ref_cvel)))
    return reward

def aslip_heightpenalty_reward(self, action):

    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    ref_pos, ref_vel = self.get_ref_state(self.phase)

    # TODO: should be variable; where do these come from?
    # TODO: see magnitude of state variables to gauge contribution to reward
    # weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]

    weight = [0.05, 0.05, 0.25, 0.25, 0.05, 0.05, 0.05, 0.25, 0.25, 0.05]

    #weight = [.1] * 10

    # mujoco state info
    joint_error = 0
    com_pos = qpos[0:3]
    lfoot_pos = self.l_foot_pos - qpos[0:3]
    rfoot_pos = self.r_foot_pos - qpos[0:3]
    com_vel = qvel[0:3]

    footpos_error     = 0
    com_vel_error     = 0
    action_penalty    = 0
    foot_orient_penalty = 0
    straight_diff = 0

    phase_to_match = self.phase + 1

    # ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_ext_state(self, self.cassie_state, self.last_pelvis_pos, phase_to_match, offset=self.vertOffset)
    ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_unaltered_state(self, phase_to_match)

    # enforce distance between feet and com
    footpos_error += np.linalg.norm(ref_lfoot - lfoot_pos) + np.linalg.norm(ref_rfoot - rfoot_pos)

    com_vel_error += np.linalg.norm(com_vel - ref_cvel)

    # each joint pos, skipping feet
    for i, j in enumerate(self.pos_idx):
        target = ref_pos[j]
        actual = qpos[j]

        if j == 20 or j == 34:
            joint_error += 0
        else:
            joint_error += (target - actual) ** 2

    # action penalty
    action_penalty = np.linalg.norm(action - self.prev_action)

    # foot orientation penalty
    foot_orient_penalty = self.l_foot_orient_cost + self.r_foot_orient_cost

    # straight difference penalty
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    
    
    reward = 0.3 * np.exp(-footpos_error) +    \
                0.3 * np.exp(-com_vel_error) +    \
                0.1 * np.exp(-action_penalty) +     \
                0.2 * np.exp(-foot_orient_penalty) + \
                0.1 * np.exp(-straight_diff)

    # apply height penalty : if less than 80% of ref_cpos[2], subtract .5 from reward
    if qpos[2] < 0.8 * ref_cpos[2]:
        reward -= 0.5
        if self.debug:
            print("Height Penality : {} < {}".format(qpos[2], 0.8 * ref_cpos[2]))

    if self.debug:
        print("reward: {10}\nfoot:\t{0:.2f}, % = {1:.2f}\ncom_vel:\t{2:.2f}, % = {3:.2f}\naction_penalty:\t{4:.2f}, % = {5:.2f}\nfoot_orient_penalty:\t{6:.2f}, % = {7:.2f}\nstraight_diff:\t{8:.2f}, % = {9:.2f}\n\n".format(
        0.3 * np.exp(-footpos_error),          0.3 * np.exp(-footpos_error) / reward * 100,
        0.3 * np.exp(-com_vel_error),          0.3 * np.exp(-com_vel_error) / reward * 100,
        0.1 * np.exp(-action_penalty),         0.1 * np.exp(-action_penalty) / reward * 100,
        0.2 * np.exp(-foot_orient_penalty),    0.2 * np.exp(-foot_orient_penalty) / reward * 100,
        0.1  * np.exp(-straight_diff),         0.1  * np.exp(-straight_diff) / reward * 100,
        reward
        )
        )
        print("actual speed: {}\tdesired_speed: {}".format(np.linalg.norm(qvel[0:3]), np.linalg.norm(ref_cvel)))
    return reward

def aslip_comorient_reward(self, action):

    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    ref_pos, ref_vel = self.get_ref_state(self.phase)

    # TODO: should be variable; where do these come from?
    # TODO: see magnitude of state variables to gauge contribution to reward
    # weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]

    weight = [0.05, 0.05, 0.25, 0.25, 0.05, 0.05, 0.05, 0.25, 0.25, 0.05]

    #weight = [.1] * 10

    # mujoco state info
    joint_error = 0
    com_pos = qpos[0:3]
    lfoot_pos = self.l_foot_pos - qpos[0:3]
    rfoot_pos = self.r_foot_pos - qpos[0:3]
    com_vel = qvel[0:3]

    footpos_error     = 0
    com_vel_error     = 0
    action_penalty    = 0
    com_orient_error  = 0
    straight_diff = 0

    phase_to_match = self.phase + 1

    # ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_ext_state(self, self.cassie_state, self.last_pelvis_pos, phase_to_match, offset=self.vertOffset)
    ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_unaltered_state(self, phase_to_match)

    # enforce distance between feet and com
    footpos_error += np.linalg.norm(ref_lfoot - lfoot_pos) + np.linalg.norm(ref_rfoot - rfoot_pos)

    com_vel_error += np.linalg.norm(com_vel - ref_cvel)

    # each joint pos, skipping feet
    for i, j in enumerate(self.pos_idx):
        target = ref_pos[j]
        actual = qpos[j]

        if j == 20 or j == 34:
            joint_error += 0
        else:
            joint_error += (target - actual) ** 2

    # action penalty
    action_penalty = np.linalg.norm(action - self.prev_action)

    # com orientation penalty
    com_orient_error = 5 * (np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0])))

    # straight difference penalty
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    
    
    reward = 0.3 * np.exp(-footpos_error) +    \
                0.3 * np.exp(-com_vel_error) +    \
                0.1 * np.exp(-action_penalty) +     \
                0.2 * np.exp(-com_orient_error) + \
                0.1 * np.exp(-straight_diff)

    if self.debug:
        print("reward: {10}\nfoot:\t{0:.2f}, % = {1:.2f}\ncom_vel:\t{2:.2f}, % = {3:.2f}\naction_penalty:\t{4:.2f}, % = {5:.2f}\ncom_orient_error:\t{6:.2f}, % = {7:.2f}\nstraight_diff:\t{8:.2f}, % = {9:.2f}\n\n".format(
        0.3 * np.exp(-footpos_error),          0.3 * np.exp(-footpos_error) / reward * 100,
        0.3 * np.exp(-com_vel_error),          0.3 * np.exp(-com_vel_error) / reward * 100,
        0.1 * np.exp(-action_penalty),         0.1 * np.exp(-action_penalty) / reward * 100,
        0.2 * np.exp(-com_orient_error),       0.2 * np.exp(-com_orient_error) / reward * 100,
        0.1  * np.exp(-straight_diff),         0.1  * np.exp(-straight_diff) / reward * 100,
        reward
        )
        )
        print("actual speed: {}\tdesired_speed: {}".format(np.linalg.norm(qvel[0:3]), np.linalg.norm(ref_cvel)))
    return reward

def aslip_comorient_heightpenalty_reward(self, action):

    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    ref_pos, ref_vel = self.get_ref_state(self.phase)

    # TODO: should be variable; where do these come from?
    # TODO: see magnitude of state variables to gauge contribution to reward
    # weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]

    weight = [0.05, 0.05, 0.25, 0.25, 0.05, 0.05, 0.05, 0.25, 0.25, 0.05]

    #weight = [.1] * 10

    # mujoco state info
    joint_error = 0
    com_pos = qpos[0:3]
    lfoot_pos = self.l_foot_pos - qpos[0:3]
    rfoot_pos = self.r_foot_pos - qpos[0:3]
    com_vel = qvel[0:3]

    footpos_error     = 0
    com_vel_error     = 0
    action_penalty    = 0
    com_orient_error  = 0
    straight_diff = 0

    phase_to_match = self.phase + 1

    # ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_ext_state(self, self.cassie_state, self.last_pelvis_pos, phase_to_match, offset=self.vertOffset)
    ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_unaltered_state(self, phase_to_match)

    # enforce distance between feet and com
    footpos_error += np.linalg.norm(ref_lfoot - lfoot_pos) + np.linalg.norm(ref_rfoot - rfoot_pos)

    com_vel_error += np.linalg.norm(com_vel - ref_cvel)

    # each joint pos, skipping feet
    for i, j in enumerate(self.pos_idx):
        target = ref_pos[j]
        actual = qpos[j]

        if j == 20 or j == 34:
            joint_error += 0
        else:
            joint_error += (target - actual) ** 2

    # action penalty
    action_penalty = np.linalg.norm(action - self.prev_action)

    # com orientation penalty
    com_orient_error = 5 * (np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0])))

    # straight difference penalty
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    
    
    reward = 0.3 * np.exp(-footpos_error) +    \
                0.3 * np.exp(-com_vel_error) +    \
                0.1 * np.exp(-action_penalty) +     \
                0.2 * np.exp(-com_orient_error) + \
                0.1 * np.exp(-straight_diff)

    # apply height penalty : if less than 80% of ref_cpos[2], subtract .5 from reward
    if qpos[2] < 0.8 * ref_cpos[2]:
        reward -= 0.5
        if self.debug:
            print("Height Penality : {} < {}".format(qpos[2], 0.8 * ref_cpos[2]))


    if self.debug:
        print("reward: {10}\nfoot:\t{0:.2f}, % = {1:.2f}\ncom_vel:\t{2:.2f}, % = {3:.2f}\naction_penalty:\t{4:.2f}, % = {5:.2f}\ncom_orient_error:\t{6:.2f}, % = {7:.2f}\nstraight_diff:\t{8:.2f}, % = {9:.2f}\n\n".format(
        0.3 * np.exp(-footpos_error),          0.3 * np.exp(-footpos_error) / reward * 100,
        0.3 * np.exp(-com_vel_error),          0.3 * np.exp(-com_vel_error) / reward * 100,
        0.1 * np.exp(-action_penalty),         0.1 * np.exp(-action_penalty) / reward * 100,
        0.2 * np.exp(-com_orient_error),       0.2 * np.exp(-com_orient_error) / reward * 100,
        0.1  * np.exp(-straight_diff),         0.1  * np.exp(-straight_diff) / reward * 100,
        reward
        )
        )
        print("actual speed: {}\tdesired_speed: {}".format(np.linalg.norm(qvel[0:3]), np.linalg.norm(ref_cvel)))
    return reward
"""