from cassie.cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis
from cassie.trajectory import CassieTrajectory
import numpy as np
import time
import matplotlib.pyplot as plt

import sys
import select
import tty
import termios
import copy

def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

sim = CassieSim("./cassie/cassiemujoco/cassie.xml")
vis = CassieVis(sim, "./cassie/cassiemujoco/cassie.xml")
u = pd_in_t()
sim.step_pd(u)

trajectory = CassieTrajectory("./cassie/trajectory/stepdata.bin")
traj = trajectory.qpos
time = 0
sim.set_qpos(traj[time, :])
sim.step_pd(u)
render_state = vis.draw(sim)

pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
mirror_obs = np.array([0.1,1,2,3,4,5,-13,-14,15,16,17,18,19,-6,-7,8,9,10,11,12])

# Make mirror matrix
numel = len(mirror_obs)
mirror_mat = np.zeros((numel, numel))

for (i, j) in zip(np.arange(numel), np.abs(mirror_obs.astype(int))):
    mirror_mat[i, j] = np.sign(mirror_obs[i])

old_settings = termios.tcgetattr(sys.stdin)
try:
    tty.setcbreak(sys.stdin.fileno())
    while render_state:
        if isData():
            c = sys.stdin.read(1)
            if c == 'm':
                curr_qpos = np.copy(sim.qpos())
                pos = curr_qpos[pos_index]
                mir_pos = pos @ mirror_mat
                mir_qpos = copy.deepcopy(curr_qpos)
                mir_qpos[pos_index] = mir_pos
                sim.set_qpos(mir_qpos)
                sim.step_pd(u)
            elif c == 'n':
                time += 100
                if time > traj.shape[0]:
                    print("Done")
                    break
                print("Time: ", time)
                sim.set_qpos(traj[time, :])
                sim.step_pd(u)
            else:
                pass
        render_state = vis.draw(sim)
        # time.sleep(0.001)
finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)