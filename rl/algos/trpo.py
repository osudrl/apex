"""Trust Region Policy Optimization"""
import numpy as np

import torch
import torch.autograd as autograd

from torch.distributions.kl import kl_divergence


# TODO: move this out and share it across algorithms
class TRPOBuffer:
    def __init__(self, gamma=0.99, lam=0.95, use_gae=False):
        self.states  = []
        self.actions = []
        self.rewards = []
        self.values  = []
        self.returns = []

        self.ep_returns = [] # for logging

        self.gamma, self.lam = gamma, lam

        self.ptr, self.path_idx = 0, 0
    
    def store(self, state, action, reward, value):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        # TODO: make sure these dimensions really make sense
        self.states  += [state.squeeze(0)]
        self.actions += [action.squeeze(0)]
        self.rewards += [reward.squeeze(0)]
        self.values  += [value.squeeze(0)]

        self.ptr += 1
    
    def finish_path(self, last_val=None):
        if last_val is None:
            last_val = np.zeros(shape=(1,))

        path = slice(self.path_idx, self.ptr)
        rewards = self.rewards[path]

        returns = []

        R = last_val.squeeze(0)
        for reward in reversed(rewards):
            R = self.gamma * R + reward
            returns.insert(0, R) # TODO: self.returns.insert(self.path_idx, R) ? 
                                 # also technically O(k^2), may be worth just reversing list
                                 # BUG? This is adding copies of R by reference (?)

        self.returns += returns

        self.ep_returns += [np.sum(rewards)]

        self.path_idx = self.ptr
    
    def get(self):
        return(
            self.states,
            self.actions,
            self.returns,
            self.values
        )

class TRPO:
    def __init__(self, args):
        self.damping = args['damping']
        self.max_kl  = args['max_kl']

        self.gamma   = args['gamma']
        self.lam     = args['lam']

        # self.lr            = args['lr']
        # self.eps           = args['eps']
        # self.entropy_coeff = args['entropy_coeff']
        # self.clip          = args['clip']
        # self.batch_size    = args['batch_size']
        # self.epochs        = args['epochs']
        # self.num_steps     = args['num_steps']

    @torch.no_grad()
    def sample(self, env, policy, min_steps, max_traj_len, deterministic=False):
        """
        Sample at least min_steps number of total timesteps, truncating 
        trajectories if they exceed max_traj_len number of timesteps
        """
        memory = TRPOBuffer(self.gamma, self.lam)

        num_steps = 0
        while num_steps < min_steps:
            state = torch.Tensor(env.reset())

            done = False
            value = 0
            traj_len = 0

            while not done and traj_len < max_traj_len:
                value, action = policy.act(state, deterministic)

                next_state, reward, done, _ = env.step(action.data.numpy())

                memory.store(state.numpy(), action.numpy(), reward, value.numpy())

                state = torch.Tensor(next_state)

                traj_len += 1
                num_steps += 1

            value, _ = policy.act(state)
            memory.finish_path(last_val=(not done) * value.numpy())
        
        return memory

    def update(policy, loss, kl, max_kl, damping):
        #loss_grad = flat(autograd.grad(loss, policy.parameters())).detach()

        stepdir = cg(fvp, -loss_grad, 10)

    def train(self, 
              env_fn, 
              policy,
              n_itr,
              normalize=None):
        
        env = Vectorize([env_fn]) # this will be useful for parallelism later
        
        if normalize is not None:
            env = normalize(env)

        old_policy = deepcopy(policy)

        optimizer = optim.Adam(policy.parameters(), lr=self.lr, eps=self.eps)

        for itr in range(n_itr):
            print("********** Iteration {} ************".format(itr))

            batch = self.sample(env, policy, self.num_steps, 400) #TODO: fix this

            observations, actions, returns, values = map(torch.Tensor, batch.get())

            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

            batch_size = self.batch_size or advantages.numel()

            print("timesteps in batch: %i" % advantages.numel())

            old_policy.load_state_dict(policy.state_dict())  # WAY faster than deepcopy

            # for _ in range(self.epochs):
            #     losses = []
            #     sampler = BatchSampler(
            #         SubsetRandomSampler(range(advantages.numel())),
            #         batch_size,
            #         drop_last=True
            #     )

            #     for indices in sampler:


""" Conjugate gradient algorithm"""
def cg(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

""" Return a flattened gradient vector"""
def flat(grads):
    return torch.cat([grad.view(-1) for grad in grads])

""" Compute the (empirical) Fisher vector product"""
def fvp(kl, pi, v, damping):
    # gradient kl div w.r.t policy params
    grads = flat(autograd.grad(kl, pi.parameters(), create_graph=True))

    # (gradient kl div w.r.t policy params) * v
    gvp = (grads * v).sum()

    # (hessian of kl div w.r.t. policy params) * v
    hvp = flat(autograd.grad(gvp, pi.parameters())).detach()

    return hvp + v * damping




