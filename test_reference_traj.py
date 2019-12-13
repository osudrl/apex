import matplotlib.pyplot as plt

import numpy as np

from cassie import CassieEnv, CassieTSEnv, CassieIKEnv, UnifiedCassieIKEnv, CassieEnv_nodelta, CassieEnv_rand_dyn, CassieEnv_speed_dfreq

# env_fn = partial(CassieIKEnv, "walking", clock_based=True, state_est=state_est, speed=speed)
env = UnifiedCassieIKEnv("walking", clock_based=True, state_est=True)


env.reset()
env.render()

qposes = []
qvels = []

# manually set the environment's speed

env.speed = 0.3

print(env.phaselen)

for i in range(env.phaselen):

    """
    output of get_ref_state for clock and state estimator
    np.concatenate([qpos[pos_index], qvel[vel_index], np.concatenate((clock, [self.speed]))]])
    """

    # get reference state
    pos, vel = env.get_ref_state(i)

    # extract qpos, qvel
    qposes.append(pos)
    qvels.append(vel)

print(len(qposes))

# plot the qpos
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.plot(qposes)
plt.savefig('ref_qposes.png')