#File to visualize realtime cassie data
import pickle
from matplotlib import pyplot as plt
import numpy as np

POLICY_NAME = "aslip_unified_0_v5"
FILE_PATH = "./hardware_logs/"
FILE_NAME = "2020-01-20_16:51_log0"

SAVE_NAME = FILE_PATH + POLICY_NAME + "/" + FILE_NAME

data = np.load(SAVE_NAME + '.npz')

# truncate data to last cycle
for key in data:
    print(key)

time = data['time']
motors = data['motor']
joints = data['joint']
torques_mea = data['torques_measured']
ff_left = data['left_foot_force']
ff_right = data['right_foot_force']
foot_pos_left = data['left_foot_pos']
foot_pos_right= data['right_foot_pos']
trajectory = data['trajectory']


row = 6
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


#Plot Trajectory Positions
ax2 = plt.subplot(row,col,2, sharex=ax1)
trajectory = np.rad2deg(trajectory)
ax2.plot(time[:], trajectory[:, 0], label='left-hip-roll'  )
# ax2.plot(time[:], trajectory[:, 1], label='left-hip-yaw'   )
ax2.plot(time[:], trajectory[:, 2], label='left-hip-pitch' )
ax2.plot(time[:], trajectory[:, 3], label='left-knee'      )
# ax2.plot(time[:], motors[:, 4], label='left-foot'      )
ax2.plot(time[:], trajectory[:, 5], label='right-hip-roll' )
# ax2.plot(time[:], trajectory[:, 6], label='right-hip-yaw'  )
ax2.plot(time[:], trajectory[:, 7], label='right-hip-pitch')
ax2.plot(time[:], trajectory[:, 8], label='right-knee'     )
# ax2.plot(time[:], trajectory[:, 9], label='right-foot'     )
# ax2.set_xlabel('Time')
ax2.set_ylabel('Traj Motor Position [deg]')
ax2.legend(loc='upper left')
ax2.set_title('Traj Motor Position')


# measured torques
ax3 = plt.subplot(row,col,3, sharex=ax1)
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
ax4 = plt.subplot(row,col,4, sharex=ax1)
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
ax5 = plt.subplot(row,col,5, sharex=ax1)
# ax5.plot(time, ff_left[:, 0], label='left-X'  )
# ax5.plot(time, ff_left[:, 1], label='left-Y'  )
ax3.plot(time, ff_left[:, 2], label='left-Z'  )
# ax3.plot(time, ff_right[:, 0], label='right-X'  )
# ax3.plot(time, ff_right[:, 1], label='right-Y'  )
ax3.plot(time, ff_right[:, 2], label='right-Z'  )
# ax3.set_xlabel('Time')
ax3.set_ylabel('Foot Force [N]')
ax3.legend(loc='upper left')
ax3.set_title('Foot Forces')

# foot force
ax5 = plt.subplot(row,col,6, sharex=ax1)
# ax5.plot(time, foot_pos_left[:, 0], label='left-X'  )
# ax5.plot(time, foot_pos_left[:, 1], label='left-Y'  )
ax5.plot(time, foot_pos_left[:, 2], label='left-Z'  )
# ax5.plot(time, foot_pos_right[:, 0], label='right-X'  )
# ax5.plot(time, foot_pos_right[:, 1], label='right-Y'  )
ax5.plot(time, foot_pos_right[:, 2], label='right-Z'  )
# ax5.set_xlabel('Time')
ax5.set_ylabel('Foot Pos [m]')
ax5.legend(loc='upper left')
ax5.set_title('Foot Pos')



# plt.tight_layout()
plt.show()