# $ ipython -i test.py

from trajectory import CassieTrajectory

traj = CassieTrajectory("stepdata.bin")

print(len(traj.qpos[0]))