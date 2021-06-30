import torch
import os
from cassie.trajectory import CassieTrajectory
from util import *

paths = ["extra/CassieClean/stand_smooth_footorient_zeroclock_stddev15_rewterm04_seed10",
        "extra/CassieClean/sprint_fromrun_trajlen300_stddev15_rewterm04_cont_purespeed_stddev20_rewterm03_seed10", 
        "running/CassieNoaccelFootDist/run_speed0-4_forcevel02_footpos07_linclock5_h010-030_cycletime1_phaseadd15_mirror4_stddev15_rewterm04_cont_stddev20_rewterm03_seed10"]
speeds = [[0], [5], [0, 0.5, 1, 1.5, 2, 2.5, 3.0, 3.5, 4.0]]

save_qpos = None
save_qvel = None

for i in range(len(paths)):
    policy = torch.load(os.path.join("./logs/", paths[i], "actor_highest.pt"))
    policy.eval()

    run_args = pickle.load(open(os.path.join("./logs/", paths[i], "experiment.pkl"), "rb"))

    env_fn = env_factory(run_args.env_name, traj=run_args.traj, simrate=run_args.simrate, clock_based=run_args.clock_based, state_est=run_args.state_est, 
                    dynamics_randomization=run_args.dyn_random, mirror=run_args.mirror, no_delta=run_args.no_delta, ik_baseline=run_args.ik_baseline, 
                    learn_gains=run_args.learn_gains, reward=run_args.reward, history=run_args.history)
    env = env_fn().env

    for j in range(len(speeds[i])):
        
        state = env.reset_for_test()
        env.update_speed(speeds[i][j])
        print("speed:", speeds[i][j], env.phase_add, env.phaselen)
        num_states = 0
        while env.counter < 2:
            action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
            state = env.step_basic(action)
        # for k in range(int(env.phaselen+1)):
        while env.counter < 3:
            action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
            for _ in range(env.simrate):
                if save_qpos is None:
                    save_qpos = env.sim.qpos()
                    save_qvel = env.sim.qvel()
                    save_qpos = np.expand_dims(save_qpos, axis=0)
                    save_qvel = np.expand_dims(save_qvel, axis=0)
                else:
                    save_qpos = np.concatenate((save_qpos, np.expand_dims(env.sim.qpos(), axis=0)), axis=0)
                    save_qvel = np.concatenate((save_qvel, np.expand_dims(env.sim.qvel(), axis=0)), axis=0)
                num_states += 1
                env.step_sim_basic(action)

            env.time  += 1
            env.phase += env.phase_add

            if env.phase > env.phaselen:
                env.phase = 0
                env.counter += 1

            state = env.get_full_state()
        print("num states: ", num_states)

old_traj = CassieTrajectory("./cassie/trajectory/stepdata.bin")
save_qpos = np.concatenate((save_qpos, old_traj.qpos), axis=0)
save_qvel = np.concatenate((save_qvel, old_traj.qvel), axis=0)

print(save_qpos.shape)
print(save_qvel.shape)
np.savez("./total_reset_states.npz", qpos=save_qpos, qvel=save_qvel)
    
