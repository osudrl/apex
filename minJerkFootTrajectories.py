
import numpy as np
import pickle
# This function returns minimum jerk foot trajectories in the form of a 2D
# evenly spaced numpy array
# Inputs:
#           initPos     initial position 3 element  array-like
#           finalPos    final position 3 element  array-like
#           nPoints     number of points that the swing trajectory will span
#           h           Peak (midpoint) Z height of the swing trajectory 
#                       relative to the initial z height
#           zdotTD      Vertical touchdown velocity normalized to length of 
#                       swing. To get this from real desired velocity, divide
#                       by the swing phase length
# Outputs:
#           out         this is a [nPoints x 4] numpy 2d array 
def min_jerk_foot_trajectories( initPos, finalPos, nPoints, h, zdotTD):

    T = 0.4
    t = np.linspace(0.0, T, nPoints)
    t1 = t[0:int(np.floor(nPoints/2))]
    t2 = t[int(np.floor(nPoints/2)):]

    xF = finalPos[0] - initPos[0]
    x = initPos[0]     \
        +(5*xF)/(2*(T**2)) * (t**2) \
        -(5*xF)/(2*(T**4)) * (t**4) \
        +xF/(T**5) * (t**5)

    yF = finalPos[1] - initPos[1]
    y = initPos[1]  \
        +(5*yF)/(2*(T**2)) * (t**2)  \
        -(5*yF)/(2*(T**4)) * (t**4)  \
        +yF/(T**5) * (t**5)

    zF = finalPos[2] - initPos[2]
    hrel = h

    z1 = initPos[2] \
        +(40*hrel + (20*zF)/3 - T*zdotTD)/(4*T**2) * (t1**2)  \
         -(40*hrel + 20*zF - 3*T*zdotTD)/T**4 * (t1**4)  \
         +(4*(24*hrel + 20*zF - 3*T*zdotTD))/(3*T**5) * (t1**5)

    z2 = initPos[2] + 2*hrel - (28*zF)/3 + (7*T*zdotTD)/4  \
        -(120*hrel - 460*zF + 87*T*zdotTD)/(6*T) * (t2)  \
        +(1080*hrel - 2860*zF + 549*T*zdotTD)/(12*T**2) * (t2**2) \
        -(480*hrel - 1040*zF + 204*T*zdotTD)/(3*T**3) * (t2**3) \
        +(360*hrel - 700*zF + 141*T*zdotTD)/(3*T**4) * (t2**4)  \
        -(4*(24*hrel - 44*zF + 9*T*zdotTD))/(3*T**5) * (t2**5)

    print(t.shape)
    print(t1.shape)
    print(t2.shape)
    print(x.shape)
    print(y.shape)
    print(z1.shape)
    print(z2.shape)

    out = np.transpose([t, x, y, np.concatenate((z1,z2)) ])

    return out



if __name__ == '__main__':

    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    initPos =[0.0,0.0,0]

    finalPos = [0.0,0.0,0]

    # traj = None
    # with open("./cassie/trajectory/foottraj_doublestance_time0.4_land0.2_h0.15.pkl", "rb") as f:
    #     traj = pickle.load(f)

    # rfoot = traj["rfoot"]
    # lfoot = traj["lfoot"]
    
    # # # com_xyz = np.copy(traj["qpos"][:, 0:3])
    # # # rfoot_relative = np.copy(traj["rfoot"])
    # # # lfoot_relative = np.copy(traj["lfoot"])
    # # # rfoot = com_xyz + rfoot_relative
    # # # lfoot = com_xyz + lfoot_relative
    # time_len = rfoot.shape[0]
    # rfoot_vel = np.zeros(time_len)
    # lfoot_vel = np.zeros(time_len)
    # for i in range(1, time_len):
    #     lfoot_vel[i] = (lfoot[i, 2] - lfoot[i-1, 2]) / (0.4 / (841 - int(1682 / 5)))
    #     rfoot_vel[i] = (rfoot[i, 2] - rfoot[i-1, 2]) / (0.4 / (841 - int(1682 / 5)))
    # save_dict = {"lfoot": lfoot, "rfoot": rfoot, "lfoot_vel": lfoot_vel, "rfoot_vel": rfoot_vel}
    # with open("./cassie/trajectory/foottraj_doublestance_time0.4_land0.2_h0.15_vels.pkl", "wb") as f:
    #     traj = pickle.dump(save_dict, f)
    # exit()

    # simrate = 1
    # phaselen = int(np.floor(time_len / simrate) - 1)
    # inds = [i*simrate for i in range(phaselen)]
    # # data = np.concatenate((lfoot[inds, 2], rfoot[]))
    # # exit()
    # rfoot_vel = np.zeros(phaselen)
    # lfoot_vel = np.zeros(phaselen)
    # for i in range(1, phaselen):
    #     lfoot_vel[i] = (lfoot[inds[i], 2] - lfoot[inds[i-1], 2]) / (0.4 / (841 - int(1682 / 5)))
    #     rfoot_vel[i] = (rfoot[inds[i], 2] - rfoot[inds[i-1], 2]) / (0.4 / (841 - int(1682 / 5)))
    # num_zero = 0
    # print(lfoot[0, 2])
    # for i in range(1, time_len):
        # if np.abs(lfoot[i-1, 2]) <= 1e-4 and num_zero < 2:
        #     print("left foot z is zero at: ", i-1)
        #     num_zero += 1
        # rfoot_vel[i] = (rfoot[i, 2] - rfoot[i-1, 2]) / (0.4 / (841 - int(1682 / 5)))
        # lfoot_vel[i] = (lfoot[i, 2] - lfoot[i-1, 2]) / (0.4 / (841 - int(1682 / 5)))
    
    # fig, ax = plt.subplots(2, 1, figsize=(15, 5), sharex=True)
    # t = np.linspace(0, 1, phaselen)
    # ax[0].plot(t, lfoot[inds, 2])
    # ax[1].plot(t, rfoot[inds, 2])
    # plt.tight_layout()
    # plt.show()
    # exit()

    # print(lfoot_vel)
    # fig, ax = plt.subplots(2, 2, figsize=(15, 5))
    # t = np.linspace(0, 1, phaselen)
    # titles = ["Foot Position", "Foot Velocity"]
    # ax[0][0].set_ylabel("z position (m)")
    # ax[1][0].set_ylabel("z velocity (m/s)")

    # ax[0][0].plot(t, lfoot[inds, 2])
    # # ax[0][0].scatter(t, lfoot[inds, 2])
    # ax[0][0].set_title("Left " + titles[0])
    # ax[1][0].plot(t, lfoot_vel[:])
    # # ax[1][0].scatter(t, lfoot_vel[:])
    # ax[1][0].set_title("Left " + titles[1])
    # ax[1][0].set_xlabel("Time (sec)")
    # ax[0][1].plot(t, rfoot[inds, 2])
    # # ax[0][1].scatter(t, rfoot[inds, 2])
    # ax[0][1].set_title("Right " + titles[0])
    # ax[1][1].plot(t, rfoot_vel[:])
    # # ax[1][1].scatter(t, rfoot_vel[:])
    # ax[1][1].set_title("Right " + titles[1])
    # ax[1][1].set_xlabel("Time (sec)")
    # plt.tight_layout()
    # plt.show()
    # exit()
    

    h = 0.2

    zdotTD = -0.4

    total_cycle_time = 1682
    double_stance_time = int(total_cycle_time / 5)
    single_step_time = int(total_cycle_time / 2) - double_stance_time

    data = min_jerk_foot_trajectories(initPos, finalPos, single_step_time, h, zdotTD)
    print("outputlen: ", data.shape[0])
    # Pad remainder of data with zeros until is length of total_cycle_time/2 for double stance time
    pad_data = np.zeros((double_stance_time, 4))
    data = np.append(data, pad_data, axis=0)
    print("data len: ", data.shape[0])
    # Copy traj for other foot and append zeros for half of traj that is not moving
    lfoot = np.append(data[:, 1:4], np.zeros((int(total_cycle_time / 2), 3)), axis = 0)
    rfoot = np.append(np.zeros((int(total_cycle_time / 2), 3)), data[:, 1:4], axis = 0)
    rfoot_vel = np.zeros(total_cycle_time)
    lfoot_vel = np.zeros(total_cycle_time)
    for i in range(1, total_cycle_time):
        lfoot_vel[i] = (lfoot[i, 2] - lfoot[i-1, 2]) / (0.4 / single_step_time)
        rfoot_vel[i] = (rfoot[i, 2] - rfoot[i-1, 2]) / (0.4 / single_step_time)
    save_dict = {"lfoot": lfoot, "rfoot": rfoot, "lfoot_vel": lfoot_vel, "rfoot_vel": rfoot_vel}
    with open("./cassie/trajectory/foottraj_doublestance_time0.4_land0.4_h0.2.pkl", "wb") as f:
        traj = pickle.dump(save_dict, f)

    print(data.shape)

    fig, ax = plt.subplots(2, 2, figsize=(15, 5))
    t = np.linspace(0, 0.4/single_step_time * total_cycle_time, total_cycle_time)
    titles = ["Foot Position", "Foot Velocity"]
    ax[0][0].set_ylabel("z position (m)")
    ax[1][0].set_ylabel("z velocity (m/s)")

    ax[0][0].plot(t, lfoot[:, 2])
    # ax[0][0].scatter(t, lfoot[inds, 2])
    ax[0][0].set_title("Left " + titles[0])
    ax[1][0].plot(t, lfoot_vel[:])
    # ax[1][0].scatter(t, lfoot_vel[:])
    ax[1][0].set_title("Left " + titles[1])
    ax[1][0].set_xlabel("Time (sec)")
    ax[0][1].plot(t, rfoot[:, 2])
    # ax[0][1].scatter(t, rfoot[inds, 2])
    ax[0][1].set_title("Right " + titles[0])
    ax[1][1].plot(t, rfoot_vel[:])
    # ax[1][1].scatter(t, rfoot_vel[:])
    ax[1][1].set_title("Right " + titles[1])
    ax[1][1].set_xlabel("Time (sec)")
    plt.tight_layout()
    plt.show()


    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot(data[:,1],data[:,2],data[:,3])
    # ax.axis('equal')
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)
    # ax.set_zlim(0,1)
    # plt.show()
    