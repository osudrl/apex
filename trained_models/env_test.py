import time 

import numpy as np

from cassie import CassieEnv

from cassie.cassiemujoco import *
from cassie.trajectory import CassieTrajectory


traj = CassieTrajectory("cassie/trajectory/stepdata-2018.bin")


env = CassieEnv("cassie/trajectory/stepdata.bin")


u = pd_in_t()
print(len(traj))
# test actual trajectory
def test_ref():
    csim = CassieSim()
    cvis = CassieVis()

    for t in range(len(traj.qpos)):
        qpos = traj.qpos[t]
        qvel = traj.qvel[t]

        csim.set_qpos(qpos)
        csim.set_qvel(qvel)

        
        y = csim.step_pd(u)


        cvis.draw(csim)

        input()

        print(t, end='\r')


# test trajectory wrap-around
def test_wrap():
    print(env.phaselen)
    while True:
        #     # start = t.time()
        #     # while True:
        #     #     stop = t.time()
        #     #     #print(stop-start)
        #     #     #print("stop")
        #     #     if stop - start > 0.033:
        #     #         break

        pos, vel = env.get_ref_state(env.phase)


        env.phase = env.phase + env.phaselen // 2 + 1
        pos2, vel2 = env.get_ref_state(env.phase)
        
        print(np.copy(pos[7:21]-pos2[21:35]).mean()) # check phase symmetry between legs

        env.phase = env.phase - env.phaselen // 2 - 1

        #print(env.sim.qpos()[0])

        env.sim.set_qpos(pos)
        env.sim.set_qvel(vel)

        y = env.sim.step_pd(u)

        #time.sleep(0.1)

        input()

        env.render()

        print(env.phase, env.counter)


        env.phase += 1

        if env.phase > env.phaselen:
            env.phase = 0
            env.counter += 1
            #break

# plot cassie state variables over time
def state_plot():
    pass
    
#test_wrap()
test_ref()
