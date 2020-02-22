import numpy as np
import matplotlib.pyplot as plt
import pickle

with open("./cassie/trajectory/foottraj_land0.4_speed1.0_fixedheightfreq_fixedtdvel.pkl", "rb") as f:
    trajectory = pickle.load(f)

rfoot = trajectory["rfoot"][:]
lfoot = trajectory["lfoot"][:]
rfoot_vel = trajectory["rfoot_vel"][:]
lfoot_vel = trajectory["lfoot_vel"][:]

data = np.load("./foottraj_land0.4_speed1.0_fixedheight_fulldata.npy")
print(data[1:4, :].shape)
# print(data[4, :])

pos_data = np.concatenate((data[1:4, :], data[4:7, :]), axis=0)
print(pos_data.shape)
vel_data = np.concatenate((data[10:13, :], data[13:16, :]), axis=0)
foot_data = np.concatenate((rfoot, lfoot), axis=1)

fig, ax = plt.subplots(2, 3)
titles = ["X", "Y", "Z"]
sides = ["Right ", "Left "]
for i in range(2):
    for j in range(3):
        ax[i][j].plot(foot_data[:, 3*i+j])
        ax[i][j].set_title(sides[i] + titles[j])

plt.tight_layout()
plt.show()