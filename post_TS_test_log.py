import pickle
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cassie
import time
from tempfile import TemporaryFile


FILE_PATH = "./testTS_logs/"
FILE_NAME = "0.5_logfinal"


logs = pickle.load(open(FILE_PATH + FILE_NAME + ".pkl", "rb")) #load in file with cassie data

# data = {"time": time_log, "output": output_log, "input": input_log, "state": state_log, "target_torques": target_torques_log,\
# "target_foot_residual": target_foot_residual_log}

time = logs["time"]
states_rl = logs["input"]
states = logs["state"]
nn_output = logs["output"]
trajectory_steps = logs["trajectory"]

numStates = len(states)
pelvis      = np.zeros((numStates, 3))
motors      = np.zeros((numStates, 10))
joints      = np.zeros((numStates, 6))
torques_mea = np.zeros((numStates, 10))
ff_left = np.zeros((numStates, 6))
ff_right = np.zeros((numStates, 6))
foot_pos_left = np.zeros((numStates, 6))
foot_pos_right = np.zeros((numStates, 6))
# trajectory_log = np.zeros((numStates, 10))

j=0
for s in states:
    pelvis[j, :] = s.pelvis.position[:]
    motors[j, :] = s.motor.position[:]
    joints[j, :] = s.joint.position[:]
    torques_mea[j, :] = s.motor.torque[:]
    ff_left[j, :] = np.reshape(np.asarray([s.leftFoot.toeForce[:],s.leftFoot.heelForce[:]]), (6))
    ff_right[j, :] = np.reshape(np.asarray([s.rightFoot.toeForce[:],s.rightFoot.heelForce[:]]), (6))
    foot_pos_left[j, :] = np.reshape(np.asarray([s.leftFoot.position[:],s.leftFoot.position[:]]), (6))
    foot_pos_right[j, :] = np.reshape(np.asarray([s.rightFoot.position[:],s.rightFoot.position[:]]), (6))
    
    # trajectory_log[j, :] = trajectory_steps[j][:]

    j += 1

# Save stuff for later
SAVE_NAME = FILE_PATH + FILE_NAME + '.npz'
# np.savez(SAVE_NAME, time = time, motor = motors, joint = joints, torques_measured=torques_mea, left_foot_force = ff_left, right_foot_force = ff_right, left_foot_pos = foot_pos_left, right_foot_pos = foot_pos_right, trajectory = trajectory_log)
np.savez(SAVE_NAME, time = time, motor = motors, joint = joints, torques_measured=torques_mea, left_foot_force = ff_left, right_foot_force = ff_right, left_foot_pos = foot_pos_left, right_foot_pos = foot_pos_right)

##########################################
# Plot everything (except for ref traj)
##########################################

row = 5
col = 1

#Plot Motor Positions
ax1 = plt.subplot(row,col,1)
motors = np.rad2deg(motors)
ax1.plot(time[:], motors[:, 0], label='left-hip-roll'  )
# ax1.plot(time[:], motors[:, 1], label='left-hip-yaw'   )
ax1.plot(time[:], motors[:, 2], label='left-hip-pitch' )
ax1.plot(time[:], motors[:, 3], label='left-knee'      )
# ax1.plot(time[:], motors[:, 4], label='left-foot'      )
ax1.plot(time[:], motors[:, 5], label='right-hip-roll' )
# ax1.plot(time[:], motors[:, 6], label='right-hip-yaw'  )
ax1.plot(time[:], motors[:, 7], label='right-hip-pitch')
ax1.plot(time[:], motors[:, 8], label='right-knee'     )
# ax1.plot(time[:], motors[:, 9], label='right-foot'     )
# ax1.set_xlabel('Time')
ax1.set_ylabel('Motor Position [deg]')
ax1.legend(loc='upper left')
ax1.set_title('Motor Position')


# measured torques
ax3 = plt.subplot(row,col,2, sharex=ax1)
ax3.plot(time, torques_mea[:, 0], label='left-hip-roll'  )
# ax3.plot(time, torques_mea[:, 1], label='left-hip-yaw'   )
ax3.plot(time, torques_mea[:, 2], label='left-hip-pitch' )
ax3.plot(time, torques_mea[:, 3], label='left-knee'      )
# ax3.plot(time, torques_mea[:, 4], label='left-foot'      )
ax3.plot(time, torques_mea[:, 5], label='right-hip-roll' )
# ax3.plot(time, torques_mea[:, 6], label='right-hip-yaw'  )
ax3.plot(time, torques_mea[:, 7], label='right-hip-pitch')
ax3.plot(time, torques_mea[:, 8], label='right-knee'     )
# ax3.plot(time, torques_mea[:, 9], label='right-foot'     )
# ax3.set_xlabel('Time')
ax3.set_ylabel('Measured Torques [Nm]')
ax3.legend(loc='upper left')
ax3.set_title('Measured Torques')

#Plot Joint Positions
ax4 = plt.subplot(row,col,3, sharex=ax1)
joints = np.rad2deg(joints)
ax4.plot(time, joints[:, 0], label='left-knee-spring'  )
# ax4.plot(time, joints[:, 1], label='left-tarsus')
# ax4.plot(time, joints[:, 2], label='left-foot'  )
ax4.plot(time, joints[:, 3], label='right-knee-spring'  )
# ax4.plot(time, joints[:, 4], label='right-tarsus')
# ax4.plot(time, joints[:, 5], label='right-foot'  )
# ax4.set_xlabel('Time')
ax4.set_ylabel('Joint Position [deg]')
ax4.legend(loc='upper left')
ax4.set_title('Joint Position')

# foot force
ax5 = plt.subplot(row,col,4, sharex=ax1)
# ax5.plot(time, ff_left[:, 0], label='left-X'  )
# ax5.plot(time, ff_left[:, 1], label='left-Y'  )
ax5.plot(time, ff_left[:, 2], label='left-Z'  )
# ax5.plot(time, ff_right[:, 0], label='right-X'  )
# ax5.plot(time, ff_right[:, 1], label='right-Y'  )
ax5.plot(time, ff_right[:, 2], label='right-Z'  )
# ax5.set_xlabel('Time')
ax5.set_ylabel('Foot Force [N]')
ax5.legend(loc='upper left')
ax5.set_title('Foot Forces')

# foot force
ax6 = plt.subplot(row,col,5, sharex=ax1)
# ax6.plot(time, foot_pos_left[:, 0], label='left-X'  )
# ax6.plot(time, foot_pos_left[:, 1], label='left-Y'  )
ax6.plot(time, foot_pos_left[:, 2], label='left-Z'  )
# ax6.plot(time, foot_pos_right[:, 0], label='right-X'  )
# ax6.plot(time, foot_pos_right[:, 1], label='right-Y'  )
ax6.plot(time, foot_pos_right[:, 2], label='right-Z'  )
# ax6.set_xlabel('Time')
ax6.set_ylabel('Foot Pos [m]')
ax6.legend(loc='upper left')
ax6.set_title('Foot Pos')

# plt.tight_layout()
# plt.show()
plt.savefig(SAVE_NAME + '_full_log.png')

# plot them side by side
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

ax.plot(pelvis[:, 0], pelvis[:, 1], pelvis[:, 2], label='pelvis')
ax.plot(pelvis[:, 0] + foot_pos_left[:, 0], pelvis[:, 1] + foot_pos_left[:, 1], pelvis[:, 2] + foot_pos_left[:, 2], label='true left foot pos')
ax.plot(pelvis[:, 0] + foot_pos_right[:, 0], pelvis[:, 1] + foot_pos_right[:, 1], pelvis[:, 2] + foot_pos_right[:, 2], label='true right foot pos')

ax.axis('equal')
plt.show()
plt.savefig(SAVE_NAME + '_TS_pose.png')