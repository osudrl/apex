import numpy as np
import imageio

# img = imageio.imread("./cassie/cassiemujoco/terrains/hfield.png")
# print(img.shape)

data = np.random.randint(255, size=(200, 250), dtype=np.uint8)
data[:, 0:8] = 0
data = np.hstack((np.flip(data, axis=1), data))
print(data.shape)
imageio.imwrite("./cassie/cassiemujoco/terrains/noisy.png", data)
exit()

x_size = 200
y_size = 200
y_pad = 0
data = np.zeros((x_size, y_size+y_pad), dtype=np.uint8)
num_cycles = 1
x = np.linspace(0, 2*np.pi*num_cycles, y_size)
wave = 255/2*(np.sin(x - np.pi/2) + 1)
# print(wave)
wave = wave.astype(np.uint8)
# print(wave)
# exit()
for i in range(y_size):
    data[:, i] = wave
    # print(data[i, :])
# print(data)

imageio.imwrite("./cassie/cassiemujoco/terrains/crown.png", data)

