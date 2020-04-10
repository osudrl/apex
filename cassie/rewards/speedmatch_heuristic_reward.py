import numpy as np

def speedmatch_heuristic_reward(self):
    ######## Pelvis z accel penalty #########
    pelaccel = np.abs(self.cassie_state.pelvis.translationalAcceleration[2])
    pelaccel_penalty = 0
    if pelaccel > 5:
        pelaccel_penalty = (pelaccel - 5) / 10
    pelbonus = 0
    if 8 < pelaccel < 10:
        pelbonus = 0.2
    ######## Foot position penalty ########
    foot_pos = np.zeros(6)
    self.sim.foot_pos(foot_pos)
    foot_dist = np.linalg.norm(foot_pos[0:2]-foot_pos[3:5])
    foot_penalty = 0
    if foot_dist < 0.14:
       foot_penalty = 0.2
    ######## Foot force penalty ########
    foot_forces = self.sim.get_foot_forces()
    lforce = max((foot_forces[0] - 350)/1000, 0)
    rforce = max((foot_forces[1] - 350)/1000, 0)
    forcebonus = 0
    # print("foot force: ", lforce, rforce)
    # lbonus = max((800 - foot_forces[0])/1000, 0)
    if foot_forces[0] <= 1000 and foot_forces[1] <= 1000:
        forcebonus = foot_forces[0] / 5000 + foot_forces[1] / 5000
    ######## Foot velocity penalty ########
    lfoot_vel_bonus = 0     
    rfoot_vel_bonus = 0
    # if self.prev_foot is not None and foot_pos[2] < 0.3 and foot_pos[5] < 0.3:
    #     lfoot_vel = np.abs(foot_pos[2] - self.prev_foot[2]) / 0.03 * 0.03
    #     rfoot_vel = np.abs(foot_pos[5] - self.prev_foot[5]) / 0.03 * 0.03
    # if self.l_high:
    #     lfoot_vel_bonus = self.lfoot_vel * 0.3
    # if self.r_high:
    #     rfoot_vel_bonus = self.rfoot_vel * 0.3
    ######## Foot orientation ########
    lfoot_orient = 1 - np.inner(np.array([1, 0, 0, 0]), self.cassie_state.leftFoot.orientation[:]) ** 2
    rfoot_orient = 1 - np.inner(np.array([1, 0, 0, 0]), self.cassie_state.rightFoot.orientation[:]) ** 2
    ####### Hip yaw ########
    rhipyaw = np.abs(qpos[22])
    lhipyaw = np.abs(qpos[8])
    if lhipyaw < 0.05:
        lhipyaw = 0
    if rhipyaw < 0.05:
        rhipyaw = 0
    ####### Hip roll penalty #########
    lhiproll = np.abs(qpos[7])
    rhiproll = np.abs(qpos[21])
    if lhiproll < 0.05:
        lhiproll = 0
    if rhiproll < 0.05:
        rhiproll = 0
    ####### Prev action penalty ########
    if self.prev_action is not None:
        prev_penalty = np.linalg.norm(self.curr_action - self.prev_action) / 10 #* (30/self.simrate)
    else:
        prev_penalty = 0

    reward = .2*np.exp(-self.com_vel_error) + .1*np.exp(-self.com_error) + .1*np.exp(-self.orientation_error) \
            + .1*np.exp(-20*self.l_foot_diff) + .1*np.exp(-5*self.l_footvel_diff) \
            + .1*np.exp(-20*self.r_foot_diff) + .1*np.exp(-5*self.r_footvel_diff) \
            + .1*np.exp(-lfoot_orient) + .1*np.exp(-rfoot_orient)
    # reward = .4*np.exp(-forward_diff) + .3*np.exp(-orient_diff) \
                # + .15*np.exp(-straight_diff) + .15*np.exp(-y_vel) \
                # + .1*np.exp(-self.l_foot_orient) + .1*np.exp(-self.r_foot_orient) \
                # + .1*np.exp(-self.smooth_cost) \
                # + .15*np.exp(-self.joint_error) 
                # + .1*np.exp(-self.torque_cost) + .1*np.exp(-self.smooth_cost) #\
                #
                #  + .075*np.exp(-10*lhipyaw) + .075*np.exp(-10*rhipyaw) + .075*np.exp(-10*lhiproll) + .075*np.exp(-10*rhiproll)
    #         + .1*np.exp(-20*self.l_foot_diff) + .1*np.exp(-20*self.r_foot_diff) \
    #         + .1*np.exp(-5*self.l_footvel_diff) + .1*np.exp(-5*self.r_footvel_diff)
    # - lfoot_vel_bonus - rfoot_vel_bonus - foot_penalty
    # - lforce - rforce
    #+ pelbonus- pelaccel_penalty - foot_penalty
