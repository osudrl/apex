import torch
import numpy as np
import tty
import termios
import select
import pickle
import sys
import time

from cassie.quaternion_function import *
from cassie import CassieEnv, CassieEnv_latent, CassieStandingEnv
from cassie.cassiemujoco.cassiemujoco import CassieSim, CassieVis
from cassie.vae import *
from torch import nn

def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def vis_policy(latent_model, norm_params, is_recurrent=False):
    # Load policy and env args
    eval_path = "./trained_models/latent_space/Cassie-v0/5b75b3-seed0/"
    run_args = pickle.load(open(eval_path + "experiment.pkl", "rb"))
    policy = torch.load(eval_path + "actor.pt")

    # Load latent model
    latent_size = 35
    hidden_size = 40
    # latent_model = VAE(hidden_size, latent_size, mj_state=True)
    # saved_state_dict = torch.load("./vae_model/mj_state_test_latent{}_hidden{}.pt".format(latent_size, hidden_size), map_location=torch.device('cpu'))
    # latent_model.load_state_dict(saved_state_dict)
    # print(latent_model)
    # exit()

    # Make interaction env and reconstruction sim/vis
    env = CassieEnv(traj=run_args.traj, state_est=run_args.state_est, dynamics_randomization=run_args.dyn_random, clock_based=run_args.clock_based, history=run_args.history)
    print("obs dim: ", env._obs)
    # NOTE: BUG!!!!! For some reason get consistent seg faults if use env.vis rather than a separate CassieVis object. Don't know why
    # For now, seems like if want to have multiple CassieVis objects need to have the actual CassieVis objects separately instead of using/creating
    # them in a CassieEnv.......... wtf
    policy_vis = CassieVis(env.sim, "./cassie/cassiemujoco/cassie.xml")
    reconstruct_sim = CassieSim("./cassie/cassiemujoco/cassie.xml")
    reconstruct_vis = CassieVis(reconstruct_sim, "./cassie/cassiemujoco/cassie.xml")
    # print("Made both env and vis")
    # print("env sim id: ", id(env.sim))
    # print("reconstruct sim id: ", id(reconstruct_sim))
    # print("reconstruct vis id: ", id(reconstruct_vis))
    # norm_params = np.load("./data_norm_params.npz")
    data_max = norm_params["data_max"][2:35]
    data_min = norm_params["data_min"][2:35]
    print("data max shape: ", data_max.shape)

    if is_recurrent:
        latent_model.reset_hidden(1)

    old_settings = termios.tcgetattr(sys.stdin)

    orient_add = 0
    perturb_duration = 0.2
    perturb_start = -100
    force_arr = np.zeros(6)
    timesteps = 0
    reconstruct_err = np.zeros(33)

    # Inital render of both vis's
    # env_render_state = env.render()
    policy_render_state = policy_vis.draw(env.sim)
    reconstruct_render_state = reconstruct_vis.draw(reconstruct_sim)
    try:
        tty.setcbreak(sys.stdin.fileno())

        state = env.reset_for_test()
        done = False
        speed = 0.0

        while policy_render_state and reconstruct_render_state:
        
            if isData():
                c = sys.stdin.read(1)
                if c == 'w':
                    speed += 0.1
                    env.update_speed(speed)
                    print("speed: ", env.speed)
                elif c == 's':
                    speed -= 0.1
                    env.update_speed(speed)
                    print("speed: ", env.speed)
                elif c == 'l':
                    orient_add += .1
                    print("Increasing orient_add to: ", orient_add)
                elif c == 'k':
                    orient_add -= .1
                    print("Decreasing orient_add to: ", orient_add)
                elif c == 'p':
                    print("set perturb time")
                    push = 20
                    push_dir = 1
                    force_arr = np.zeros(6)
                    force_arr[push_dir] = push
                    # env.sim.apply_force(force_arr)
                    perturb_start = env.sim.time()
            
            # If model is reset (pressing backspace while in vis window) then need to reset
            # perturb_start as well
            if env.sim.time() == 0:
                perturb_start = -100

            if (not policy_vis.ispaused()) and (not reconstruct_vis.ispaused()):
                # Update Orientation
                quaternion = euler2quat(z=orient_add, y=0, x=0)
                iquaternion = inverse_quaternion(quaternion)

                if env.state_est:
                    curr_orient = state[1:5]
                    curr_transvel = state[14:17]
                else:
                    curr_orient = state[2:6]
                    curr_transvel = state[20:23]
                
                new_orient = quaternion_product(iquaternion, curr_orient)

                if new_orient[0] < 0:
                    new_orient = -new_orient

                new_translationalVelocity = rotate_by_quaternion(curr_transvel, iquaternion)
                
                if env.state_est:
                    state[1:5] = torch.FloatTensor(new_orient)
                    state[14:17] = torch.FloatTensor(new_translationalVelocity)
                    # state[0] = 1      # For use with StateEst. Replicate hack that height is always set to one on hardware.
                else:
                    state[2:6] = torch.FloatTensor(new_orient)
                    state[20:23] = torch.FloatTensor(new_translationalVelocity)
                    
                # Apply perturb if needed
                if env.sim.time() - perturb_start < perturb_duration:
                    policy_vis.apply_force(force_arr, "cassie-pelvis")

                with torch.no_grad():
                    action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
                state, reward, done, _ = env.step(action)
                
                # Update reconstruct state
                curr_qpos = env.sim.qpos()
                # mj_state = env.sim.qpos()#np.concatenate([env.sim.qpos(), env.sim.qvel()])
                
                norm_state = np.divide((np.array(env.sim.qpos()[2:]) - data_min), data_max)
                if is_recurrent:
                    norm_state = torch.Tensor(norm_state.reshape(1, 1, -1))
                else:
                    norm_state = torch.Tensor(norm_state)
                # norm_state = np.random.rand(35)
                decode_state, mu, logvar = latent_model.forward(norm_state)
                decode_state = latent_model.decode(mu).detach().numpy()[0]
                # print("decode state: ", decode_state)
                if is_recurrent:
                    decode_state = decode_state[0, :]
                reconstruct_state = (decode_state*data_max) + data_min
                reconstruct_state = np.concatenate([curr_qpos[0:2], reconstruct_state])
                # reconstruct_state += data_min
                # print("reconstruct state: ", reconstruct_state)
                # print("mj state: ", mj_state)
                # print("mu: ", mu)
                # reconstruct_state[3:7] = mj_state[3:7]
                reconstruct_sim.set_qpos(reconstruct_state[0:35])
                # reconstruct_sim.set_qvel(reconstruct_state[35:35+32])
                reconstruct_err += norm_state.data.numpy().reshape(33) - decode_state

                timesteps += 1

            # Render both env and recosntruct_vis
            # render_state = env.render()
            policy_render_state = policy_vis.draw(env.sim)
            reconstruct_vis.draw(reconstruct_sim)
            time.sleep(0.02)


    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    print("Average reconstruction error: ", np.linalg.norm(reconstruct_err) / timesteps)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # print("recon shape:", recon_x.shape)
    # print("x shape: ", x.flatten().shape)
    recon_loss_cri = nn.MSELoss()

    MSE = 35 * recon_loss_cri(recon_x, x.view(-1, 35))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD


def vis_traj(latent_model, norm_params):
    data = np.load("./5b75b3-seed0_full_mjdata.npz")
    state_data = data["total_data"]
    state_data = state_data[:, 0:-32]
    print("state data std dev: ", np.std(state_data, axis=0))
    data_len =  state_data.shape[0]
    # norm_params = np.load("./data_norm_params_qpos_entropyloss.npz")
    data_max = norm_params["data_max"]
    data_min = norm_params["data_min"]
    norm_data = np.divide((state_data-data_min), data_max)
    norm_data = torch.Tensor(norm_data)


    # Load latent model
    latent_size = 35
    hidden_size = 40
    # latent_model = VAE(hidden_size, latent_size, mj_state=True)
    # saved_state_dict = torch.load("./vae_model/mj_state_qpos_entropyloss_latent{}_hidden{}.pt".format(latent_size, hidden_size), map_location=torch.device('cpu'))
    # latent_model.load_state_dict(saved_state_dict)

    decode, mu, log_var = latent_model.forward(norm_data)
    print("log_var shape: ", torch.mean(log_var, axis=0))
    print("mu std dev: ", np.std(mu.detach().numpy(), axis=0))
    loss = loss_function(decode, norm_data, mu, log_var)
    print("loss: ", loss/data_len)
    recon_data = decode.data.numpy()*data_max + data_min
    percent_error = np.divide((recon_data-state_data), state_data)
    print("avg percent error: ", np.mean(percent_error, axis=0))
    print("recon data var: ", np.std(recon_data, axis=0))

    num_trajs = state_data.shape[0] // 300
    print("num_trajs: ", num_trajs)

    # Build both sims and vis
    sim = CassieSim("./cassie/cassiemujoco/cassie.xml")
    vis = CassieVis(sim, "./cassie/cassiemujoco/cassie.xml")
    reconstruct_sim = CassieSim("./cassie/cassiemujoco/cassie.xml")
    reconstruct_vis = CassieVis(reconstruct_sim, "./cassie/cassiemujoco/cassie.xml")

    render_state = vis.draw(sim)
    recon_render_state = reconstruct_vis.draw(reconstruct_sim)

    done = False
    while (not done) and render_state and recon_render_state:
        traj_ind = input("Render what trajectory? (between 0 and {}) ".format(num_trajs-1))
        if traj_ind == "q":
            done = True
            continue
        update_t = time.time()
        timestep = 0
        traj_ind = int(traj_ind)
        while render_state and recon_render_state and timestep < 300:
            if (not vis.ispaused()) and (not reconstruct_vis.ispaused()) and time.time() - update_t >= 1/30:
                curr_qpos = state_data[300*traj_ind+timestep, :]
                recon_qpos = recon_data[300*traj_ind+timestep, :]
                sim.set_qpos(curr_qpos)
                reconstruct_sim.set_qpos(recon_qpos)
                timestep += 1
                update_t = time.time()
            vis.draw(sim)
            reconstruct_vis.draw(reconstruct_sim)
            time.sleep(0.005)

@torch.no_grad()
def interpolate_latent(latent_model, norm_params):
    def isData():
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
    old_settings = termios.tcgetattr(sys.stdin)

    ind1 = 0
    ind2 = 15
    data = np.load("./5b75b3-seed0_full_mjdata.npz")
    state_data = data["total_data"]
    state_data = state_data[:, 0:-32]
    data_len =  state_data.shape[0]
    data_max = norm_params["data_max"]
    data_min = norm_params["data_min"]
    norm_data = np.divide((state_data-data_min), data_max)
    norm_data = torch.Tensor(norm_data)
    decode1, mu1, logvar1 = latent_model.forward(norm_data[ind1, :])
    decode2, mu2, logvar2 = latent_model.forward(norm_data[ind2, :])

    num_points = 20
    mu_diff = mu2 - mu1
    mu_interp = [mu1+(i/num_points)*mu_diff for i in range(num_points+1)]

    sim = CassieSim("./cassie/cassiemujoco/cassie.xml")
    vis = CassieVis(sim, "./cassie/cassiemujoco/cassie.xml")
    # print("decode mu: ", latent_model.decode(mu1))
    recon_state = latent_model.decode(mu1)[0].detach().numpy()
    recon_state = np.multiply(recon_state,data_max) + data_min
    sim.set_qpos(recon_state)
    render_state = vis.draw(sim)
    mu_ind = 0
    try:
        tty.setcbreak(sys.stdin.fileno())
        while render_state:
            if isData():
                c = sys.stdin.read(1)
                if c == 'w':
                    mu_ind += 1
                    mu_ind = min(num_points, mu_ind)
                    recon_state = latent_model.decode(mu_interp[mu_ind])[0].detach().numpy()
                    recon_state = np.multiply(recon_state,data_max) + data_min
                    sim.set_qpos(recon_state)
                    print("mu_ind: ", mu_ind)
                elif c == 's':
                    mu_ind -= 1
                    mu_ind = max(0, mu_ind)
                    recon_state = latent_model.decode(mu_interp[mu_ind])[0].detach().numpy()
                    recon_state = np.multiply(recon_state,data_max) + data_min
                    sim.set_qpos(recon_state)
                    print("mu_ind: ", mu_ind)
                elif c == 'a':
                    recon_state = latent_model.decode(mu1)[0].detach().numpy()
                    recon_state = np.multiply(recon_state,data_max) + data_min
                    sim.set_qpos(recon_state)
                    print("Setting to mu1 state")
                elif c == 'd':
                    recon_state = latent_model.decode(mu2)[0].detach().numpy()
                    recon_state = np.multiply(recon_state,data_max) + data_min
                    sim.set_qpos(recon_state)
                    print("Setting to mu2 state")
                elif c == 'q':
                    sim.set_qpos(state_data[ind1, :])
                    print("Setting to orig state 1")
                elif c == 'e':
                    sim.set_qpos(state_data[ind2, :])
                    print("Setting to orig state 2")
            
            render_state = vis.draw(sim)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


# Load latent model
latent_size = 25
hidden_size = 64
layer = 1
input_dim = 33
# latent_model = VAE(hidden_size=hidden_size, latent_size = latent_size, input_size = input_dim, mj_state=True)
# latent_model = RNN_VAE_FULL(hidden_size=hidden_size, latent_size=latent_size, num_layers=layer, input_size=input_dim,  device="cpu", mj_state=True)
# latent_model = VAE_LSTM_FF(hidden_size=hidden_size, latent_size=latent_size, num_layers=layer, input_size=input_dim,  device="cpu", mj_state=True)
# latent_model = VAE_LSTM_FF_Relu(hidden_size=hidden_size, latent_size=latent_size, num_layers=layer, input_size=input_dim, device="cpu", mj_state=True)
latent_model = VAE_LSTM_FF_Relu_less(hidden_size=hidden_size, latent_size=latent_size, num_layers=layer, input_size=input_dim, device="cpu", mj_state=True)

# saved_state_dict = torch.load("./vae_model/mj_state_SSE_KL_NoXY_500epoch_latent_{}_hidden_{}.pt".format(latent_size, hidden_size), map_location=torch.device('cpu'))
saved_state_dict = torch.load("./vae_model/mj_state_lstm_FF_Relu_less_SSE_randInit_2layer32_NoKL_NoXY_latent25_layer1_hidden64.pt", map_location=torch.device('cpu'))

latent_model.load_state_dict(saved_state_dict)
norm_params = np.load("./total_mjdata_norm_params.npz")


# vis_traj(latent_model, norm_params)
# vis_policy(latent_model, norm_params, is_recurrent=False)
vis_policy(latent_model, norm_params, is_recurrent=True)
# interpolate_latent(latent_model, norm_params)