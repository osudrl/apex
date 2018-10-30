# import numpy as np

# from cassie_env import CassieEnv

# from mujoco.cassiemujoco import *
# from trajectory.trajectory import CassieTrajectory


# traj = CassieTrajectory("trajectory/stepdata.bin")


# env = CassieEnv("trajectory/stepdata.bin")
# csim = CassieSim()

# u = pd_in_t()

# test actual trajectory

# for i in range(len(traj.qpos)):
#     qpos = traj.qpos[i]
#     qvel = traj.qvel[i]

#     csim.set_qpos(qpos)
#     csim.set_qvel(qvel)

#     y = csim.step_pd(u)

#     cvis.draw(csim)

#     print(i, end='\r')


# test trajectory wrap-around

# env.render()
# env.reset()

# u = pd_in_t()
# while True:
#     # start = t.time()
#     # while True:
#     #     stop = t.time()
#     #     #print(stop-start)
#     #     #print("stop")
#     #     if stop - start > 0.033:
#     #         break

#     pos, vel = env.get_ref_state()

#     '''env.phase = env.phase + 14
#     pos2, vel2 = env.get_kin_state()
#     print(pos[7:21]-pos2[21:35])
#     env.phase = env.phase - 14'''

#     env.phase += 1
#     # #print(env.speed)
#     if env.phase >= 28:
#         env.phase = 0
#         env.counter += 1
#         #break
#     env.sim.set_qpos(pos)
#     env.sim.set_qvel(vel)
#     y = env.sim.step_pd(u)
#     env.render()