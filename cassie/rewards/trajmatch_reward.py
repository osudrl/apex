import numpy as np

def trajmatch_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    phase_diff = self.phase - np.floor(self.phase)
    ref_pos_prev, ref_vel_prev = self.get_ref_state(int(np.floor(self.phase)))
    if phase_diff != 0:
        ref_pos_next, ref_vel_next = self.get_ref_state(int(np.ceil(self.phase)))
        ref_pos_diff = ref_pos_next - ref_pos_prev
        ref_vel_diff = ref_vel_next - ref_vel_prev
        ref_pos = ref_pos_prev + phase_diff*ref_pos_diff
        ref_vel = ref_vel_prev + phase_diff*ref_vel_diff
    else:
        ref_pos = ref_pos_prev
        ref_vel = ref_vel_prev

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

    # orientation error does not look informative
    # maybe because it's comparing euclidean distance on quaternions
    # print("reward: {8}\njoint:\t{0:.2f}, % = {1:.2f}\ncom:\t{2:.2f}, % = {3:.2f}\norient:\t{4:.2f}, % = {5:.2f}\nspring:\t{6:.2f}, % = {7:.2f}\n\n".format(
    #             0.5 * np.exp(-joint_error),       0.5 * np.exp(-joint_error) / reward * 100,
    #             0.3 * np.exp(-com_error),         0.3 * np.exp(-com_error) / reward * 100,
    #             0.1 * np.exp(-orientation_error), 0.1 * np.exp(-orientation_error) / reward * 100,
    #             0.1 * np.exp(-spring_error),      0.1 * np.exp(-spring_error) / reward * 100,
    #             reward
    #         )
    #     )

    return reward

def trajmatch_footorient_hiprollvelact_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    phase_diff = self.phase - np.floor(self.phase)
    ref_pos_prev, ref_vel_prev = self.get_ref_state(int(np.floor(self.phase)))
    if phase_diff != 0:
        ref_pos_next, ref_vel_next = self.get_ref_state(int(np.ceil(self.phase)))
        ref_pos_diff = ref_pos_next - ref_pos_prev
        ref_vel_diff = ref_vel_next - ref_vel_prev
        ref_pos = ref_pos_prev + phase_diff*ref_pos_diff
        ref_vel = ref_vel_prev + phase_diff*ref_vel_diff
    else:
        ref_pos = ref_pos_prev
        ref_vel = ref_vel_prev

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

    reward = 0.3 * np.exp(-joint_error) +       \
             0.2 * np.exp(-com_error) +         \
             0.1 * np.exp(-orientation_error) + \
             0.1 * np.exp(-spring_error) \
            + .075*np.exp(-self.l_foot_orient_cost) + .075*np.exp(-self.r_foot_orient_cost) \
            + .1*np.exp(-self.hiproll_cost) + 0.05*np.exp(-self.hiproll_act)

    # orientation error does not look informative
    # maybe because it's comparing euclidean distance on quaternions
    # print("reward: {8}\njoint:\t{0:.2f}, % = {1:.2f}\ncom:\t{2:.2f}, % = {3:.2f}\norient:\t{4:.2f}, % = {5:.2f}\nspring:\t{6:.2f}, % = {7:.2f}\n\n".format(
    #             0.5 * np.exp(-joint_error),       0.5 * np.exp(-joint_error) / reward * 100,
    #             0.3 * np.exp(-com_error),         0.3 * np.exp(-com_error) / reward * 100,
    #             0.1 * np.exp(-orientation_error), 0.1 * np.exp(-orientation_error) / reward * 100,
    #             0.1 * np.exp(-spring_error),      0.1 * np.exp(-spring_error) / reward * 100,
    #             reward
    #         )
    #     )

    return reward