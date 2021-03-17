import numpy as np
from cassie.cassiemujoco import CassieSim, CassieVis
import time

reset_states = np.load("./total_reset_states.npz")
qpos = reset_states["qpos"]
qvel = reset_states["qvel"]

sim = CassieSim("./cassie/cassiemujoco/cassie.xml")
vis = CassieVis(sim, "./cassie/cassiemujoco/cassie.xml")

render_state = vis.draw(sim)
curr_idx = 0

while render_state and curr_idx < qpos.shape[0]:
    if (not vis.ispaused()):
        print(f"curr idx: {curr_idx}", end="\r")
        sim.set_qpos(qpos[curr_idx, :])
        curr_idx += 1

    render_state = vis.draw(sim)
    time.sleep(0.0005)



