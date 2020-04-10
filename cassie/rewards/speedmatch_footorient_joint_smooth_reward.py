import numpy as np

def speedmatch_footorient_joint_smooth_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    
    orient_targ = np.array([1, 0, 0, 0])
    speed_targ = np.array([self.speed, 0, 0])
    if self.time >= self.orient_time:
        orient_targ = euler2quat(z=self.orient_add, y=0, x=0)
        iquaternion = inverse_quaternion(orient_targ)
        speed_targ = rotate_by_quaternion(speed_targ, iquaternion)
        new_orient = quaternion_product(iquaternion, self.cassie_state.pelvis.orientation[:])
        if new_orient[0] < 0:
            new_orient = -new_orient
    forward_diff = np.abs(qvel[0] - speed_targ[0])
    orient_diff = 1 - np.inner(orient_targ, qpos[3:7]) ** 2
    # orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1] - speed_targ[1])
    if forward_diff < 0.05:
        forward_diff = 0
    if y_vel < 0.05:
        y_vel = 0
    straight_diff = 8*np.abs(qpos[1] - self.y_offset)
    if np.abs(qpos[1] - self.y_offset) < 0.05:
        straight_diff = 0
    if orient_diff < 5e-3:
        orient_diff = 0
    else:
        orient_diff *= 30

    reward = .25*np.exp(-forward_diff) + .1*np.exp(-orient_diff) \
                + .1*np.exp(-straight_diff) + .1*np.exp(-y_vel) \
                + .1*np.exp(-self.l_foot_orient) + .1*np.exp(-self.r_foot_orient) \
                + .1*np.exp(-self.smooth_cost) \
                + .15*np.exp(-self.joint_error) 

    return reward