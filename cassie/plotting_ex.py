import numpy as np
import time
import math

# from cassie_env import CassieEnv

from cassiemujoco import *
from trajectory.trajectory import CassieTrajectory
import matplotlib.pyplot as plt 
from matplotlib import style
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython import display

def visualise_sim_graph(file_path, freq_of_sim):
    traj = np.load(file_path)
    # env = CassieEnv("walking")
    # csim = CassieSim("./cassie/cassiemujoco/cassie.xml")
    # vis = CassieVis(csim, "./cassie/cassiemujoco/cassie.xml")
    u = pd_in_t()

    # pelvisXYZ = traj.f.qpos_replay[:, 0:3]
    # render_state = vis.draw(csim)
    # saved_time = traj.f.time[:]

    #################Graphing###########
    log_time = traj.f.time[:]
    y_val = traj.f.qpos_replay[:,2] #z - height
    x_data= log_time
    y_data = y_val

    delt_x = (x_data[1] - x_data[0]) * 1000 #convert seconds to ms

    num_frames = math.ceil(len(x_data) / 10)



    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    output = plt.plot([])
    plt.close()
    print(output[0])

    x = np.linspace(0,2*np.pi, 100)

    fig = plt.figure()

    lines = plt.plot([])
    line = lines[0]

    #other setup //set x and y lims
    plt.xlim(x_data.min(), x_data.max())
    plt.ylim(y_data.min(), y_data.max())
    def animate(frame):
        #update
        x = x_data[:frame*10]
        y = y_data[:frame*10]
        # y = np.sin(x + 2*np.pi * frame/100)
        line.set_data((x,y))

    anim = FuncAnimation(fig, animate, frames=num_frames, interval=(1/freq_of_sim * 1000 + (10 * delt_x))) #20 is 50 fps

    anim.save('lines.mp4', writer=writer)
    # html = display.HTML(video)
    # display.display(html)

    plt.close()

visualise_sim_graph("./outfile8.npz", 30)