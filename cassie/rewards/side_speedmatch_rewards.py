import numpy as np

def side_speedmatch_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    
    forward_diff = np.abs(qvel[0] -self.speed)
    orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    side_diff = np.abs(qvel[1] - self.side_speed)
    if forward_diff < 0.05:
        forward_diff = 0
    if side_diff < 0.05:
        side_diff = 0

    reward = .4*np.exp(-forward_diff) + .4*np.exp(-side_diff) + .2*np.exp(-orient_diff)

    return reward

def side_speedmatch_torquesmooth_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    
    forward_diff = np.abs(qvel[0] -self.speed)
    orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    side_diff = np.abs(qvel[1] - self.side_speed)
    if forward_diff < 0.05:
        forward_diff = 0
    if side_diff < 0.05:
        side_diff = 0

    reward = .25*np.exp(-forward_diff) + .25*np.exp(-side_diff) + .2*np.exp(-orient_diff) \
                + .1*np.exp(-self.torque_cost) + .2*np.exp(-self.smooth_cost)
                
    return reward

def side_speedmatch_foottraj_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    
    forward_diff = np.abs(qvel[0] -self.speed)
    orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    side_diff = np.abs(qvel[1] - self.side_speed)
    if forward_diff < 0.05:
        forward_diff = 0
    if side_diff < 0.05:
        side_diff = 0

    reward = .15*np.exp(-forward_diff) + .15*np.exp(-side_diff) + .1*np.exp(-orient_diff) \
                    + .1*np.exp(-20*self.l_foot_diff) + .1*np.exp(-20*self.r_foot_diff) \
                    + .1*np.exp(-5*self.l_footvel_diff) + .1*np.exp(-5*self.r_footvel_diff) \
                    + .1*np.exp(-self.lfoot_orient_cost) + .1*np.exp(-self.rfoot_orient_cost)

    return reward

def side_speedmatch_heightvel_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    
    forward_diff = np.abs(qvel[0] -self.speed)
    orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    side_diff = np.abs(qvel[1] - self.side_speed)
    if forward_diff < 0.05:
        forward_diff = 0
    if side_diff < 0.05:
        side_diff = 0

    reward = .2*np.exp(-forward_diff) + .2*np.exp(-side_diff) + .1*np.exp(-orient_diff) \
            + .1*np.exp(-self.lfoot_orient_cost) + .1*np.exp(-self.rfoot_orient_cost) \
            + .15*np.exp(-self.lf_heightvel) + .15*np.exp(-self.rf_heightvel) \
            # + .1*np.exp(-self.ltdvel_cost) * .1*np.exp(-self.rtdvel_cost)

    return reward

def side_speedmatch_heuristic_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    
    forward_diff = np.abs(qvel[0] -self.speed)
    orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    side_diff = np.abs(qvel[1] - self.side_speed)
    if forward_diff < 0.05:
        forward_diff = 0
    if side_diff < 0.05:
        side_diff = 0

    ######## Foot position penalty ########
    foot_pos = np.zeros(6)
    self.sim.foot_pos(foot_pos)
    foot_dist = np.linalg.norm(foot_pos[0:2]-foot_pos[3:5])
    foot_penalty = 0
    if foot_dist < 0.22:
       foot_penalty = 0.2
    ######## Foot force penalty ########
    foot_forces = self.sim.get_foot_forces()
    lforce = max((foot_forces[0] - 700)/1000, 0)
    rforce = max((foot_forces[1] - 700)/1000, 0)
    ######## Torque penalty ########
    torque = np.linalg.norm(self.cassie_state.motor.torque[:])        
    ######## Pelvis z accel penalty #########
    pelaccel = np.abs(self.cassie_state.pelvis.translationalAcceleration[2])
    pelaccel_penalty = 0
    if pelaccel > 6:
        pelaccel_penalty = (pelaccel - 6) / 30
    ####### Prev action penalty ########
    if self.prev_action is not None:
        prev_penalty = np.linalg.norm(self.curr_action - self.prev_action) / 10 #* (30/self.simrate)
    else:
        prev_penalty = 0
    print("prev_penalty: ", prev_penalty)
    ######## Foot height bonus ########
    footheight_penalty = 0
    if (np.abs(self.lfoot_vel) < 0.05 and foot_pos[2] < 0.2 and foot_forces[0] == 0) or (np.abs(self.rfoot_vel) < 0.05 and foot_pos[5] < 0.2 and foot_forces[1] == 0):
        # print("adding foot height penalty")
        footheight_penalty = 0.2


    reward = .25*np.exp(-forward_diff) + .25*np.exp(-side_diff) + .1*np.exp(-orient_diff) \
            + .1*np.exp(-self.torque_cost) + .1*np.exp(-self.smooth_cost) \
            + .1*np.exp(-self.lfoot_orient_cost) + .1*np.exp(-self.rfoot_orient_cost) \
            - pelaccel_penalty \
            - foot_penalty \
            - lforce - rforce \
            - footheight_penalty

    return reward