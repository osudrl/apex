import math
import numpy as np

def quaternion2euler(quaternion):
	w = quaternion[0]
	x = quaternion[1]
	y = quaternion[2]
	z = quaternion[3]
	ysqr = y * y
	
	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + ysqr)
	X = math.degrees(math.atan2(t0, t1))
	
	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	Y = math.degrees(math.asin(t2))
	
	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (ysqr + z * z)
	Z = math.degrees(math.atan2(t3, t4))

	result = np.zeros(3)
	result[0] = X * np.pi / 180
	result[1] = Y * np.pi / 180
	result[2] = Z * np.pi / 180
	
	return result

def euler2quat(z=0, y=0, x=0):

    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    result =  np.array([
             cx*cy*cz - sx*sy*sz,
             cx*sy*sz + cy*cz*sx,
             cx*cz*sy - sx*cy*sz,
             cx*cy*sz + sx*cz*sy])
    if result[0] < 0:
    	result = -result
    return result

def command_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    # get current speed and orientation
    curr_pos = qpos[0:3]
    curr_speed = qvel[0]
    curr_orient = quaternion2euler(qpos[3:7])[2]

    # desired speed and orientation
    desired_pos    = self.command_traj.global_pos[self.command_counter] + self.last_position
    desired_speed  = self.command_traj.speed_cmd[self.command_counter]
    desired_orient = self.command_traj.orient[self.command_counter]

    compos_error      = np.linalg.norm(curr_pos - desired_pos)
    speed_error       = np.linalg.norm(curr_speed - desired_speed)
    orientation_error = np.linalg.norm(curr_orient - desired_orient)

    reward = 0.2 * np.exp(-speed_error) +       \
             0.3 * np.exp(-compos_error) +       \
             0.5 * np.exp(-orientation_error)

    if self.debug:
        print("reward: {6}\nspeed:\t{0:.2f}, % = {1:.2f}\ncompos:\t{2:.2f}, % = {3:.2f}\norient:\t{4:.2f}, % = {5:.2f}\n\n".format(
        0.325  * np.exp(-speed_error),       0.325 * np.exp(-speed_error) / reward * 100,
        0.35 * np.exp(-compos_error),    0.35 * np.exp(-compos_error) / reward * 100,
        0.325 * np.exp(-orientation_error),    0.325 * np.exp(-orientation_error) / reward * 100,
        reward
        )
        )
        print(self.command_counter)
        print("actual speed:  {}\tdesired_speed:  {}".format(curr_speed, self.speed))
        print("actual compos: {}\tdesired_pos:    {}".format(curr_pos[0:2], desired_pos[0:2]))
        print("actual orient: {}\tdesired_orient: {}".format(curr_orient, desired_orient))
    return reward

def command_reward_no_pos(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    # get current speed and orientation
    # curr_pos = qpos[0:3]
    curr_speed = qvel[0]
    curr_orient = quaternion2euler(qpos[3:7])[2]

    # desired speed and orientation
    desired_speed  = self.command_traj.speed_cmd[self.command_counter]
    desired_orient = self.command_traj.orient[self.command_counter]

    # compos_error      = np.linalg.norm(curr_pos - desired_pos)
    speed_error       = np.linalg.norm(curr_speed - desired_speed)
    orientation_error = np.linalg.norm(curr_orient - desired_orient)

    reward = 0.5 * np.exp(-speed_error) +       \
             0.5 * np.exp(-orientation_error)

    if self.debug:
        print("reward: {4}\nspeed:\t{0:.2f}, % = {1:.2f}\norient:\t{2:.2f}, % = {3:.2f}\n\n".format(
        0.5  * np.exp(-speed_error),       0.5 * np.exp(-speed_error) / reward * 100,
        0.5 * np.exp(-orientation_error),    0.5 * np.exp(-orientation_error) / reward * 100,
        reward
        )
        )
        print(self.command_counter)
        print("actual speed:  {}\tdesired_speed:  {}".format(curr_speed, self.speed))
        # print("actual compos: {}\tdesired_pos:    {}".format(curr_pos[0:2], desired_pos[0:2]))
        print("actual orient: {}\tdesired_orient: {}".format(curr_orient, desired_orient))
    return reward

def command_reward_keepalive(self):
    reward = 1.0
    if self.debug:
        print("reward = 1.0\tcounter={}".format(self.command_counter))
    return reward