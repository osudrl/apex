"""Proximal Policy Optimization (clip objective)."""
from copy import deepcopy

import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import kl_divergence

from torch.nn.utils.rnn import pad_sequence

import time

import numpy as np
import os, sys

import ray

from rl.envs import WrapEnv

from rl.policies.actor import Gaussian_FF_Actor, Gaussian_LSTM_Actor
from rl.policies.critic import FF_V, LSTM_V
from rl.envs.normalize import get_normalization_params, PreNormalizer
from .ppo import PPOBuffer

import pickle

# O(n)
def merge_buf(buffers):
    merged = PPOBuffer(buffers[0].gamma, buffers[0].lam)
    for buf in buffers:
        offset = len(merged)

        merged.states  += buf.states
        merged.actions += buf.actions
        merged.rewards += buf.rewards
        merged.values  += buf.values
        merged.returns += buf.returns

        merged.ep_returns += buf.ep_returns
        merged.ep_lens    += buf.ep_lens

        merged.traj_idx += [offset + i for i in buf.traj_idx[1:]]
        merged.ptr += buf.ptr

    return merged


@ray.remote
class sample_worker:
    def __init__(self, env_fn, policy, critic, gamma, lam, model=None):
        self.env = WrapEnv(env_fn)
        if model is not None:
            self.env.remake_cassie(model)
        self.gamma = gamma
        self.lam = lam
        self.policy = policy
        self.critic = critic

    def sync_policy(self, new_actor_params, new_critic_params):
        """
        Function to sync the actor and critic parameters with new parameters.
        Args:
            new_actor_params (torch dictionary): New actor parameters to copy over
            new_critic_params (torch dictionary): New critic parameters to copy over
            input_norm (int): Running counter of states for normalization 
        """
        for p, new_p in zip(self.policy.parameters(), new_actor_params):
            p.data.copy_(new_p)

        for p, new_p in zip(self.critic.parameters(), new_critic_params):
            p.data.copy_(new_p)

    @torch.no_grad()
    def sample(self, min_steps, max_traj_len, deterministic=False, anneal=1.0, term_thresh=0):
        """
        Sample at least min_steps number of total timesteps, truncating
        trajectories only if they exceed max_traj_len number of timesteps
        """
        torch.set_num_threads(1)    # By default, PyTorch will use multiple cores to speed up operations.
                                    # This can cause issues when Ray also uses multiple cores, especially on machines
                                    # with a lot of CPUs. I observed a significant speedup when limiting PyTorch 
                                    # to a single core - I think it basically stopped ray workers from stepping on each
                                    # other's toes.

        memory = PPOBuffer(self.gamma, self.lam)

        num_steps = 0
        while num_steps < min_steps:
            state = torch.Tensor(self.env.reset())

            done = False
            value = 0
            traj_len = 0

            if hasattr(self.policy, 'init_hidden_state'):
                self.policy.init_hidden_state()

            if hasattr(self.critic, 'init_hidden_state'):
                self.critic.init_hidden_state()

            while not done and traj_len < max_traj_len:
                action = self.policy(state, deterministic=False, anneal=anneal)
                value = self.critic(state)

                next_state, reward, done, _ = self.env.step(action.numpy(), term_thresh=term_thresh)

                memory.store(state.numpy(), action.numpy(), reward, value.numpy())

                state = torch.Tensor(next_state)

                traj_len += 1
                num_steps += 1

            value = self.critic(state)
            memory.finish_path(last_val=(not done) * value.numpy())

        return memory

class PPO_modelsamp:
    def __init__(self, args, actor, critic, env_fn, save_path, model_rand = False):
        self.env_name       = args['env_name']
        self.gamma          = args['gamma']
        self.lam            = args['lam']
        self.lr             = args['lr']
        self.eps            = args['eps']
        self.entropy_coeff  = args['entropy_coeff']
        self.clip           = args['clip']
        self.minibatch_size = args['minibatch_size']
        self.epochs         = args['epochs']
        self.num_steps      = args['num_steps']
        self.max_traj_len   = args['max_traj_len']
        self.use_gae        = args['use_gae']
        self.n_proc         = args['num_procs']
        self.grad_clip      = args['max_grad_norm']
        self.recurrent      = args['recurrent']

        self.total_steps = 0
        self.highest_reward = -1
        self.past500_reward = -1
        self.limit_cores = 0

        self.save_path = save_path
        self.actor = actor
        self.critic = critic
        self.old_policy = deepcopy(actor)
        
        if model_rand:
            model_list = ["cassie.xml", "cassie_tray_box.xml", "cassie_cart_soft.xml", "cassie_carry_pole.xml", "cassie_jug_spring.xml"]
            num_model_work = self.n_proc // len(model_list)
            print("Number of workers per model:", num_model_work)
            self.samplers = [sample_worker.remote(env_fn, actor, critic, self.gamma, self.lam, model_list[i]) for _ in range(num_model_work) for i in range(len(model_list))]
        else:
            self.samplers = [sample_worker.remote(env_fn, actor, critic, self.gamma, self.lam) for _ in range(self.n_proc)]

    def save(self, policy, critic):

        try:
            os.makedirs(self.save_path)
        except OSError:
            pass
        filetype = ".pt" # pytorch model
        torch.save(policy, os.path.join(self.save_path, "actor" + filetype))
        torch.save(critic, os.path.join(self.save_path, "critic" + filetype))

    def save_cutoff(self, policy, critic, cutoff):

        try:
            os.makedirs(self.save_path)
        except OSError:
            pass
        filetype = ".pt" # pytorch model
        torch.save(policy, os.path.join(self.save_path, "actor_{}".format(cutoff) + filetype))
        torch.save(critic, os.path.join(self.save_path, "critic_{}".format(cutoff) + filetype))

    def update_policy(self, obs_batch, action_batch, return_batch, advantage_batch, mask, env_fn, mirror_observation=None, mirror_action=None):
        policy = self.actor
        critic = self.critic
        old_policy = self.old_policy

        values = critic(obs_batch)
        pdf = policy.distribution(obs_batch)

        # TODO, move this outside loop?
        with torch.no_grad():
            old_pdf = old_policy.distribution(obs_batch)
            old_log_probs = old_pdf.log_prob(action_batch).sum(-1, keepdim=True)

        log_probs = pdf.log_prob(action_batch).sum(-1, keepdim=True)

        ratio = (log_probs - old_log_probs).exp()

        cpi_loss = ratio * advantage_batch * mask
        clip_loss = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage_batch * mask
        actor_loss = -torch.min(cpi_loss, clip_loss).mean()

        critic_loss = 0.5 * ((return_batch - values) * mask).pow(2).mean()

        entropy_penalty = -(self.entropy_coeff * pdf.entropy() * mask).mean()

        # Mirror Symmetry Loss
        if mirror_observation is not None and mirror_action is not None:
            env = env_fn()
            deterministic_actions = policy(obs_batch)
            if env.clock_based:
                if self.recurrent:
                    mir_obs = torch.stack([mirror_observation(obs_batch[i,:,:], env.clock_inds) for i in range(obs_batch.shape[0])])
                    mirror_actions = policy(mir_obs)
                else:
                    mir_obs = mirror_observation(obs_batch, env.clock_inds)
                    mirror_actions = policy(mir_obs)
            else:
                if self.recurrent:
                    mirror_actions = policy(mirror_observation(torch.stack([mirror_observation(obs_batch[i,:,:]) for i in range(obs_batch.shape[0])])))
                else:
                    mirror_actions = policy(mirror_observation(obs_batch))
            mirror_actions = mirror_action(mirror_actions)
            mirror_loss = 4 * (deterministic_actions - mirror_actions).pow(2).mean()
        else:
            mirror_loss = 0

        self.actor_optimizer.zero_grad()
        (actor_loss + mirror_loss + entropy_penalty).backward()

        # Clip the gradient norm to prevent "unlucky" minibatches from
        # causing pathological updates
        torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # Clip the gradient norm to prevent "unlucky" minibatches from
        # causing pathological updates
        torch.nn.utils.clip_grad_norm_(critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        with torch.no_grad():
            kl = kl_divergence(pdf, old_pdf)

        if mirror_observation is not None and mirror_action is not None:
            mirror_loss_return = mirror_loss.item()
        else:
            mirror_loss_return = 0
        return actor_loss.item(), pdf.entropy().mean().item(), critic_loss.item(), ratio.mean().item(), kl.mean().item(), mirror_loss_return

    def train(self,
              env_fn,
              n_itr,
              logger=None, anneal_rate=1.0):

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.eps)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, eps=self.eps)

        start_time = time.time()

        env = env_fn()
        obs_mirr, act_mirr = None, None
        if hasattr(env, 'mirror_observation'):
            if env.clock_based:
                obs_mirr = env.mirror_clock_observation
            else:
                obs_mirr = env.mirror_observation

        if hasattr(env, 'mirror_action'):
            act_mirr = env.mirror_action

        curr_anneal = 1.0
        curr_thresh = 0
        start_itr = 0
        ep_counter = 0
        do_term = False
        rew_thresh = [50, 100, 150, 200, 250, 300]
        for itr in range(n_itr):
            if itr % 500 == 0:
                self.past500_reward = -1

            print("********** Iteration {} ************".format(itr))

            sample_start = time.time()
            if self.highest_reward > (2/3)*self.max_traj_len and curr_anneal > 0.5:
                curr_anneal *= anneal_rate
            if do_term and curr_thresh < 0.5:
                # break
                curr_thresh += (0.5) / 8000#.1 * 1.0006**(itr-start_itr)
            # batch = self.sample_parallel(env_fn, self.policy, self.critic, self.num_steps, self.max_traj_len, anneal=curr_anneal, term_thresh=curr_thresh)
            buffers = ray.get([w.sample.remote(self.num_steps // self.n_proc, self.max_traj_len, False, curr_anneal, curr_thresh) for w in self.samplers])
            batch = merge_buf(buffers)

            print("time elapsed: {:.2f} s".format(time.time() - start_time))
            samp_time = time.time() - sample_start
            print("sample time elapsed: {:.2f} s".format(samp_time))

            observations, actions, returns, values = map(torch.Tensor, batch.get())

            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

            minibatch_size = self.minibatch_size or advantages.numel()

            print("timesteps in batch: %i" % advantages.numel())
            self.total_steps += advantages.numel()

            self.old_policy.load_state_dict(self.actor.state_dict())

            optimizer_start = time.time()
            
            for epoch in range(self.epochs):
                losses = []
                entropies = []
                kls = []
                if self.recurrent:
                    random_indices = SubsetRandomSampler(range(len(batch.traj_idx)-1))
                    sampler = BatchSampler(random_indices, minibatch_size, drop_last=False)
                else:
                    random_indices = SubsetRandomSampler(range(advantages.numel()))
                    sampler = BatchSampler(random_indices, minibatch_size, drop_last=True)

                for indices in sampler:
                    if self.recurrent:
                        obs_batch       = [observations[batch.traj_idx[i]:batch.traj_idx[i+1]] for i in indices]
                        action_batch    = [actions[batch.traj_idx[i]:batch.traj_idx[i+1]] for i in indices]
                        return_batch    = [returns[batch.traj_idx[i]:batch.traj_idx[i+1]] for i in indices]
                        advantage_batch = [advantages[batch.traj_idx[i]:batch.traj_idx[i+1]] for i in indices]
                        mask            = [torch.ones_like(r) for r in return_batch]

                        obs_batch       = pad_sequence(obs_batch, batch_first=False)
                        action_batch    = pad_sequence(action_batch, batch_first=False)
                        return_batch    = pad_sequence(return_batch, batch_first=False)
                        advantage_batch = pad_sequence(advantage_batch, batch_first=False)
                        mask            = pad_sequence(mask, batch_first=False)
                    else:
                        obs_batch       = observations[indices]
                        action_batch    = actions[indices]
                        return_batch    = returns[indices]
                        advantage_batch = advantages[indices]
                        mask            = 1

                    scalars = self.update_policy(obs_batch, action_batch, return_batch, advantage_batch, mask, env_fn, mirror_observation=obs_mirr, mirror_action=act_mirr)
                    actor_loss, entropy, critic_loss, ratio, kl, mirror_loss = scalars

                    entropies.append(entropy)
                    kls.append(kl)
                    losses.append([actor_loss, entropy, critic_loss, ratio, kl, mirror_loss])

                # TODO: add verbosity arguments to suppress this
                print(' '.join(["%g"%x for x in np.mean(losses, axis=0)]))

                # Early stopping
                if np.mean(kl) > 0.02:
                    print("Max kl reached, stopping optimization early.")
                    break

            actor_param_id  = ray.put(list(self.actor.parameters()))
            critic_param_id = ray.put(list(self.critic.parameters()))
            ray.get([w.sync_policy.remote(actor_param_id, critic_param_id) for w in self.samplers])
            opt_time = time.time() - optimizer_start
            print("optimizer time elapsed: {:.2f} s".format(opt_time))

            # if np.mean(batch.ep_lens) >= self.max_traj_len * 0.5:
            if np.mean(batch.ep_returns) >= rew_thresh[start_itr]:
                ep_counter += 1
            # if do_term == False and ep_counter > 20:
            #     # do_term = True
            #     # start_itr = itr
            #     self.save_cutoff(policy, critic, rew_thresh[start_itr])
            #     start_itr += 1
            #     start_itr = min(start_itr, len(rew_thresh))
            #     ep_counter = 0

            if logger is not None:
                evaluate_start = time.time()
                buffers = ray.get([w.sample.remote((self.num_steps // 2) // self.n_proc, self.max_traj_len, True) for w in self.samplers])
                test = merge_buf(buffers)
                eval_time = time.time() - evaluate_start
                print("evaluate time elapsed: {:.2f} s".format(eval_time))

                avg_eval_reward = np.mean(test.ep_returns)
                avg_batch_reward = np.mean(batch.ep_returns)
                avg_ep_len = np.mean(batch.ep_lens)
                mean_losses = np.mean(losses, axis=0)
                # print("avg eval reward: {:.2f}".format(avg_eval_reward))

                sys.stdout.write("-" * 37 + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Return (test)', avg_eval_reward) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Return (batch)', avg_batch_reward) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Mean Eplen', avg_ep_len) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Mean KL Div', "%8.3g" % kl) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Mean Entropy', "%8.3g" % entropy) + "\n")
                sys.stdout.write("-" * 37 + "\n")
                sys.stdout.flush()

                entropy = np.mean(entropies)
                kl = np.mean(kls)

                logger.add_scalar("Test/Return", avg_eval_reward, itr)
                logger.add_scalar("Train/Return", avg_batch_reward, itr)
                logger.add_scalar("Train/Mean Eplen", avg_ep_len, itr)
                logger.add_scalar("Train/Mean KL Div", kl, itr)
                logger.add_scalar("Train/Mean Entropy", entropy, itr)

                logger.add_scalar("Misc/Critic Loss", mean_losses[2], itr)
                logger.add_scalar("Misc/Actor Loss", mean_losses[0], itr)
                logger.add_scalar("Misc/Mirror Loss", mean_losses[5], itr)
                logger.add_scalar("Misc/Timesteps", self.total_steps, itr)

                logger.add_scalar("Misc/Sample Times", samp_time, itr)
                logger.add_scalar("Misc/Optimize Times", opt_time, itr)
                logger.add_scalar("Misc/Evaluation Times", eval_time, itr)
                logger.add_scalar("Misc/Sample Rate", advantages.numel() / samp_time, itr)
                logger.add_scalar("Misc/Termination Threshold", curr_thresh, itr)

            # TODO: add option for how often to save model
            if self.highest_reward < avg_eval_reward:
                self.highest_reward = avg_eval_reward
                self.save_cutoff(self.actor, self.critic, "highest")

            if self.past500_reward < avg_eval_reward:
                self.past500_reward = avg_eval_reward
                self.save(self.actor, self.critic)

def run_experiment(args):
    from util import env_factory, create_logger

    # torch.set_num_threads(1)

    if args.ik_baseline and args.no_delta:
        args.ik_baseline = False

    # TODO: remove this at some point once phase_based is stable
    if args.phase_based:
        args.clock_based = False

    # wrapper function for creating parallelized envs
    env_fn = env_factory(args.env_name, traj=args.traj, simrate=args.simrate, phase_based=args.phase_based, clock_based=args.clock_based, state_est=args.state_est, no_delta=args.no_delta, learn_gains=args.learn_gains, ik_baseline=args.ik_baseline, dynamics_randomization=args.dyn_random, mirror=args.mirror, reward=args.reward, history=args.history)
    obs_dim = env_fn().observation_space.shape[0]
    action_dim = env_fn().action_space.shape[0]

    # Set up Parallelism
    os.environ['OMP_NUM_THREADS'] = '1'
    if not ray.is_initialized():
        if args.redis_address is not None:
            ray.init(num_cpus=args.num_procs, redis_address=args.redis_address)
        else:
            ray.init(num_cpus=args.num_procs)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.previous is not None:
        policy = torch.load(os.path.join(args.previous, "actor.pt"))
        critic = torch.load(os.path.join(args.previous, "critic.pt"))
        # TODO: add ability to load previous hyperparameters, if this is something that we event want
        # with open(args.previous + "experiment.pkl", 'rb') as file:
        #     args = pickle.loads(file.read())
        print("loaded model from {}".format(args.previous))
    else:
        if args.recurrent:
            policy = Gaussian_LSTM_Actor(obs_dim, action_dim, fixed_std=np.exp(-2), env_name=args.env_name)
            critic = LSTM_V(obs_dim)
        else:
            if args.learn_stddev:
                policy = Gaussian_FF_Actor(obs_dim, action_dim, fixed_std=None, env_name=args.env_name, bounded=args.bounded)
            else:
                policy = Gaussian_FF_Actor(obs_dim, action_dim, fixed_std=np.exp(args.std_dev), env_name=args.env_name, bounded=args.bounded)
            critic = FF_V(obs_dim)

        if args.do_prenorm:
            with torch.no_grad():
                policy.obs_mean, policy.obs_std = map(torch.Tensor, get_normalization_params(iter=args.input_norm_steps, noise_std=1, policy=policy, env_fn=env_fn, procs=args.num_procs))
        else:
            policy.obs_mean = torch.zeros(obs_dim)
            policy.obs_std = torch.ones(obs_dim)
        critic.obs_mean = policy.obs_mean
        critic.obs_std = policy.obs_std
    
    policy.train()
    critic.train()

    print("obs_dim: {}, action_dim: {}".format(obs_dim, action_dim))

    # create a tensorboard logging object
    logger = create_logger(args)

    algo = PPO_modelsamp(args=vars(args), actor=policy, critic=critic, env_fn=env_fn, save_path=logger.dir, model_rand=args.model_rand)

    print()
    print("Environment: {}".format(args.env_name))
    print(" ├ traj:           {}".format(args.traj))
    print(" ├ phase_based:    {}".format(args.phase_based))
    print(" ├ clock_based:    {}".format(args.clock_based))
    print(" ├ state_est:      {}".format(args.state_est))
    print(" ├ dyn_random:     {}".format(args.dyn_random))
    print(" ├ no_delta:       {}".format(args.no_delta))
    print(" ├ mirror:         {}".format(args.mirror))
    print(" ├ ik baseline:    {}".format(args.ik_baseline))
    print(" ├ learn gains:    {}".format(args.learn_gains))
    print(" ├ reward:         {}".format(env_fn().reward_func))
    print(" └ obs_dim:        {}".format(obs_dim))

    print()
    print("Synchronous Distributed Proximal Policy Optimization:")
    print(" ├ recurrent:      {}".format(args.recurrent))
    print(" ├ run name:       {}".format(args.run_name))
    print(" ├ max traj len:   {}".format(args.max_traj_len))
    print(" ├ seed:           {}".format(args.seed))
    print(" ├ num procs:      {}".format(args.num_procs))
    print(" ├ lr:             {}".format(args.lr))
    print(" ├ eps:            {}".format(args.eps))
    print(" ├ lam:            {}".format(args.lam))
    print(" ├ gamma:          {}".format(args.gamma))
    print(" ├ learn stddev:  {}".format(args.learn_stddev))
    print(" ├ std_dev:        {}".format(args.std_dev))
    print(" ├ entropy coeff:  {}".format(args.entropy_coeff))
    print(" ├ clip:           {}".format(args.clip))
    print(" ├ minibatch size: {}".format(args.minibatch_size))
    print(" ├ epochs:         {}".format(args.epochs))
    print(" ├ num steps:      {}".format(args.num_steps))
    print(" ├ use gae:        {}".format(args.use_gae))
    print(" ├ max grad norm:  {}".format(args.max_grad_norm))
    print(" └ max traj len:   {}".format(args.max_traj_len))
    print()

    algo.train(env_fn, args.n_itr, logger=logger, anneal_rate=args.anneal)
