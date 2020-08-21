import numpy as np

def jonah_RNN_reward(self):
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

        joint_error += 50 * weight[i] * (target - actual) ** 2

    # center of mass: x, y, z
    for j in [0, 1, 2]:
        target = ref_pos[j]
        actual = qpos[j]

        # NOTE: in Xie et al y target is 0

        com_error += 10 * (target - actual) ** 2

    actual_q = qpos[3:7]
    target_q = ref_pos[3:7]
    #target_q = [1, 0, 0, 0]
    orientation_error = 5 * (1 - np.inner(actual_q, target_q) ** 2)

    # left and right shin springs
    for i in [15, 29]:
        target = ref_pos[i] # NOTE: in Xie et al spring target is 0
        actual = qpos[i]

        spring_error += 1000 * (target - actual) ** 2      

    reward = 0.200 * np.exp(-joint_error) +       \
            0.450 * np.exp(-com_error) +         \
            0.300 * np.exp(-orientation_error) + \
            0.050 * np.exp(-spring_error)

    return reward