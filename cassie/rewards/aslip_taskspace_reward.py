import numpy as np

def get_ref_footdist(self, phase=None):

    if phase is None:
        phase = self.phase

    if phase > self.phaselen:
        phase = 0

    rpos = np.copy(self.trajectory.rpos[phase])
    lpos = np.copy(self.trajectory.lpos[phase])

    return rpos, lpos

def get_ref_com_vel(self, phase=None):

    if phase is None:
        phase = self.phase

    if phase > self.phaselen:
        phase = 0

    cvel = np.copy(self.trajectory.cvel[phase])

    return cvel

def get_ref_aslip_ext_state(self, phase=None, offset=None):

    if phase is None:
        phase = self.phase

    if phase > self.phaselen:
        phase = 0

    phase = int(phase)

    rpos = np.copy(self.trajectory.rpos[phase])
    rvel = np.copy(self.trajectory.rvel[phase])
    lpos = np.copy(self.trajectory.lpos[phase])
    lvel = np.copy(self.trajectory.lvel[phase])
    cpos = np.copy(self.trajectory.cpos[phase])
    cvel = np.copy(self.trajectory.cvel[phase])

    # Manual z offset to get taller walking
    if offset is not None:
        cpos[2] += offset
        # need to update these because they 
        lpos[2] -= offset
        rpos[2] -= offset

    return rpos, rvel, lpos, lvel, cpos, cvel

def aslip_reward(self, action):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    ref_pos, ref_vel = self.get_ref_state(self.phase)

    # TODO: should be variable; where do these come from?
    # TODO: see magnitude of state variables to gauge contribution to reward
    # weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]

    # weight = [0.05, 0.05, 0.25, 0.25, 0.05, 
    #           0.05, 0.05, 0.25, 0.25, 0.05]

    #weight = [.1] * 10

    joint_error          = 0
    footpos_error        = 0
    com_vel_error        = 0
    action_penalty       = 0
    foot_orient_penalty  = 0

    # enforce distance between feet and com
    ref_rfoot, ref_lfoot  = self.get_ref_footdist(self.phase + 1)

    # left foot
    lfoot = self.cassie_state.leftFoot.position[:]
    rfoot = self.cassie_state.rightFoot.position[:]
    for j in [0, 1, 2]:
        footpos_error += np.linalg.norm(lfoot[j] - ref_lfoot[j]) +  np.linalg.norm(rfoot[j] - ref_rfoot[j])
    
    if self.debug:
        print("ref_rfoot: {}  rfoot: {}".format(ref_rfoot, rfoot))
        print("ref_lfoot: {}  lfoot: {}".format(ref_lfoot, lfoot))
        print(footpos_error)

    # try to match com velocity
    ref_cvel = self.get_ref_com_vel(self.phase + 1)

    # center of mass vel: x, y, z
    cvel = self.cassie_state.pelvis.translationalVelocity
    for j in [0, 1, 2]:
        com_vel_error += np.linalg.norm(cvel[j] - ref_cvel[j])

    # each joint pos, skipping feet
    for i, j in enumerate(self.reward_pos_idx):
        target = ref_pos[j]
        actual = qpos[j]

        if j == 20 or j == 34:
            joint_error += 0
        else:
            joint_error += (target - actual) ** 2

    # action penalty
    action_penalty = np.linalg.norm(action - self.prev_action)

    # foot orientation penalty
    foot_orient_penalty = self.l_foot_orient + self.r_foot_orient

    reward = 0.3 * np.exp(-joint_error) +       \
                0.175 * np.exp(-footpos_error) +    \
                0.175 * np.exp(-com_vel_error) +    \
                0.175 * np.exp(-action_penalty) +     \
                0.175 * np.exp(-foot_orient_penalty)

    if self.debug:
        print("reward: {10}\njoint:\t{0:.2f}, % = {1:.2f}\nfoot:\t{2:.2f}, % = {3:.2f}\ncom_vel:\t{4:.2f}, % = {5:.2f}\naction_penalty:\t{6:.2f}, % = {7:.2f}\nfoot_orient_penalty:\t{8:.2f}, % = {9:.2f}\n\n".format(
        0.3  * np.exp(-joint_error),       0.3 * np.exp(-joint_error) / reward * 100,
        0.175 * np.exp(-footpos_error),    0.175 * np.exp(-footpos_error) / reward * 100,
        0.175 * np.exp(-com_vel_error),    0.175 * np.exp(-com_vel_error) / reward * 100,
        0.175 * np.exp(-action_penalty),       0.175 * np.exp(-action_penalty) / reward * 100,
        0.175 * np.exp(-foot_orient_penalty), 0.175 * np.exp(-foot_orient_penalty) / reward * 100,
        reward
        )
        )
        print("actual speed: {}\tdesired_speed: {}".format(qvel[0], self.speed))
    return reward

def aslip_TaskSpace_reward(self, action):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    
    phase_to_match = self.phase + 1

    # offset now directly in trajectories
    ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_ext_state(self, phase_to_match, offset=0.1)

    footpos_error        = 0
    compos_error         = 0
    com_vel_error        = 0
    foot_orient_penalty  = 0

    # enforce distance between feet and com
    lfoot = self.cassie_state.leftFoot.position[:]
    rfoot = self.cassie_state.rightFoot.position[:]
    footpos_error += np.linalg.norm(lfoot - ref_lfoot)
    # for j in [0, 1, 2]:
    #     footpos_error += np.linalg.norm(lfoot[j] - ref_lfoot[j]) +  np.linalg.norm(rfoot[j] - ref_rfoot[j])
    
    # enforce com position matching
    com_pos = self.cassie_state.pelvis.position[:]
    com_pos[2] -= self.cassie_state.terrain.height
    compos_error += np.linalg.norm(ref_cpos - com_pos)
    # for j in [0, 1, 2]:
    #     compos_error += np.linalg.norm(ref_cpos[j] - com_pos[j])
    
    if self.debug:
        print("ref_rfoot: {}  rfoot: {}".format(ref_rfoot, rfoot))
        print("ref_lfoot: {}  lfoot: {}".format(ref_lfoot, lfoot))
        print(footpos_error)
        print("ref_cpos:  {}   cpos: {}".format(ref_cpos, com_pos))
        print(compos_error)

    # try to match com velocity
    cvel = self.cassie_state.pelvis.translationalVelocity
    com_vel_error += np.linalg.norm(cvel - ref_cvel)
    # for j in [0, 1, 2]:
    #     com_vel_error += np.linalg.norm(cvel[j] - ref_cvel[j])

    # foot orientation penalty
    foot_orient_penalty = self.l_foot_orient + self.r_foot_orient

    reward = 0.2 * np.exp(-footpos_error) +    \
             0.2 * np.exp(-footpos_error) +    \
             0.4 * np.exp(-com_vel_error) +    \
             0.2 * np.exp(-foot_orient_penalty)

    if True:
        print("reward: {8}\nfoot:\t{0:.2f}, % = {1:.2f}\ncom_pos:\t{2:.2f}, % = {3:.2f}\ncom_vel:\t{4:.2f}, % = {5:.2f}\nfoot_orient_penalty:\t{6:.2f}, % = {7:.2f}\n\n".format(
        0.2 * np.exp(-footpos_error),          0.4 * np.exp(-footpos_error) / reward * 100,
        0.2 * np.exp(-compos_error),           0.3 * np.exp(-compos_error) / reward * 100,
        0.4 * np.exp(-com_vel_error),          0.4 * np.exp(-com_vel_error) / reward * 100,
        0.2 * np.exp(-foot_orient_penalty),    0.2 * np.exp(-foot_orient_penalty) / reward * 100,
        reward
        )
        )
        print("actual speed: {}\tdesired_speed: {}".format(qvel[0], self.speed))
    return reward

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