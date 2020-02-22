from cassie.cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis
from cassie.trajectory import CassieTrajectory
import numpy as np
import time
import matplotlib.pyplot as plt

sim = CassieSim("./cassie/cassiemujoco/cassie.xml")
vis = CassieVis(sim, "./cassie/cassiemujoco/cassie.xml")
u = pd_in_t()
sim.step_pd(u)
render_state = vis.draw(sim)
# traj = np.load("./iktraj_land0.4_speed1.0_fixedheightfreq_fixedtdvel.npy")
trajectory = CassieTrajectory("./cassie/trajectory/stepdata.bin")
traj = trajectory.qpos
start_ind = 0
# traj[0:200, 20] += 0.2
# traj[200:250, 20] += 0.1
# traj[300:350, 20] -= 0.1
# traj[350:400, 20] -= 0.15
# traj[400:500, 20] -= 0.25
# traj[500:550, 20] -= 0.3
# traj[550:600, 20] -= 0.35
# traj[600:650, 20] -= 0.38
# traj[650:700, 20] -= 0.35
# traj[700:800, 20] -= 0.3
# traj[800:850, 20] -= 0.25
# traj[850:900, 20] -= 0.2
# traj[900:950, 20] -= 0.15
# traj[950:1050, 20] -= 0.1
# traj[1050:1150, 20] -= 0.05
# traj[1380:1450, 20] += 0.05
# traj[1450:1550, 20] += 0.1
# traj[1550:1630, 20] += 0.15
# traj[1630:, 20] += 0.2

# traj[950:1050, 20] -= 0.35
# traj[1050:1150, 20] -= 0.25
# traj[1150:1250, 20] -= 0.15
# traj[1250:1450, 20] -= 0.05
# traj[1600:, 20] += 0.05


# traj[0:80, 34] -= 0.22
# traj[80:160, 34] -= 0.15
# traj[160:220, 34] -= 0.1
# traj[220:280, 34] -= 0.05
# traj[550:700, 34] += 0.05
# traj[700:770, 34] += 0.1
# traj[770:850, 34] += 0.15
# traj[850:950, 34] += 0.2
# traj[950:1050, 34] += 0.1
# traj[1050:1150, 34] += 0.05
# traj[1250:1300, 34] -= 0.1
# traj[1300:1350, 34] -= 0.2
# traj[1350:1400, 34] -= 0.3
# traj[1400:1600, 34] -= 0.35
# traj[1600:1650, 34] -= 0.3
# traj[1650:, 34] -= 0.25
# traj[1600:, 34] -= 0.4
# traj[1200:1450, 34] -= 0.05
# traj[1450:, 34] -= 0.1
# traj = np.concatenate((np.zeros((traj.shape[0], 1)), traj, np.zeros((traj.shape[0], 32)), np.zeros((traj.shape[0], 30))), axis=1)
# print(traj.shape)
# np.save("./iktraj_land0.4_speed1.0_fixedheightfreq_fixedtdvel_fixedfoot.npy", traj)
# exit()

# cassie_traj = CassieTrajectory("./cassie/trajectory/stepdata.bin")
# traj = cassie_traj.qpos

curr_foot = np.zeros(6)
foot_pos = np.zeros((traj.shape[0], 6))
foot_vel = np.zeros((traj.shape[0], 6))

print(traj.shape)
i = start_ind
while render_state:
    if (not vis.ispaused()) and (start_ind <= i < traj.shape[0]):
# for i in range(0, traj.shape[0]):
        print(i)
        
        sim.set_qpos(traj[i, :])
        sim.step_pd(u)
        sim.foot_pos(curr_foot)
        foot_pos[i, :] = curr_foot
        if i > 0:
            foot_vel[i, :] = (foot_pos[i, :] - foot_pos[i-1, :]) / 0.0005
        
        i += 1
    render_state = vis.draw(sim)
    time.sleep(0.005)

exit()
fig, ax = plt.subplots(2, 3)
titles = ["X", "Y", "Z"]
sides = ["Left ", "Right "]
for i in range(2):
    for j in range(3):
        ax[i][j].plot(foot_pos[:, 3*i+j])
        ax[i][j].set_title(sides[i] + titles[j])

plt.tight_layout()
plt.show()
# print(foot_vel[600:750, 5])
exit()

# fig, ax = plt.subplots(2, 3)
# titles = ["X", "Y", "Z"]
# sides = ["Left ", "Right "]
# for i in range(2):
#     for j in range(3):
#         ax[i][j].plot(foot_vel[:, 3*i+j])
#         ax[i][j].set_title(sides[i] + titles[j])

# plt.tight_layout()
# plt.show()
