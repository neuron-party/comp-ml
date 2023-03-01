# phasic policy gradient
# phasic policy gradient
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PPG:
    def __init__(self, observation_space, action_space, model, **params):
        '''
        Example ReplayBuffer setup for N_pi = 3, num_steps = 4, and 2 vectorized environments

          N_pi = 1                               N_pi = 2                               N_pi = 3
        [ env_1, obs t    ,   env_2 obs t      | env_1, obs t    ,   env_2 obs t      | env_1, obs t    ,   env_2 obs t      ]
        [ env_1, obs t + 1,   env_2 obs t + 1  | env_1, obs t + 1,   env_2 obs t + 1  | env_1, obs t + 1,   env_2 obs t + 1  ]
        [ env_1, obs t + 2,   env_2 obs t + 2  | env_1, obs t + 2,   env_2 obs t + 2  | env_1, obs t + 2,   env_2 obs t + 2  ]
        [ env_1, obs t + 3,   env_2 obs t + 3  | env_1, obs t + 3,   env_2 obs t + 3  | env_1, obs t + 3,   env_2 obs t + 3  ]

        '''
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.gamma = params['gamma']
        self.lambd = params['lambd']
        self.lr = params['lr']
        self.epsilon = params['epsilon']
        self.vf_clip = params['vf_clip']
        self.max_grad_norm = params['max_grad_norm']
        self.batch_norm_adv = params['batch_norm_adv']
        self.c1 = params['c1']
        self.c2 = params['c2']
        
        self.device = params['device']
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.num_steps = params['num_steps']
        self.num_epochs = params['num_epochs']
        self.num_batches = params['num_batches']
        self.num_envs = params['num_envs']
        self.batch_size = self.num_envs * self.num_steps // self.num_batches
        self.num_updates = 0
        
        self.n_pi = params['n_pi']   # number of policy phases before an auxiliary phase
        self.e_aux = params['e_aux'] # number of gradient updates during auxiliary phase
        self.num_aux_rollouts = params['num_aux_rollouts']
        self.beta_clone = params['beta_clone']
        # official paper uses e_pi = v_pi = 1 so keeping the same gradient update for PPO
        
        # storage setup
        self.states = torch.zeros((self.num_steps, self.num_envs, *self.observation_space)).to(self.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.log_probs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.value_estimates = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        
        # i think we can save a lot of gpu memory by storing these on the cpu 
        self.auxiliary_states = np.zeros((self.num_steps, self.num_envs * self.n_pi, *self.observation_space))
        self.auxiliary_value_targets = np.zeros((self.num_steps, self.num_envs * self.n_pi))
        self.auxiliary_pi = np.zeros((self.num_steps, self.num_envs * self.n_pi, self.action_space))
        self.n_pi_index = 0 # range of this index should be [0, num_envs * n_pi]
        
        
    def get_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            pi, pi_logits, value, _ = self.model(state)
            action = pi.sample()
            log_probs = pi.log_prob(action)
            
        return action.detach().cpu().numpy(), log_probs.detach().cpu().numpy(), value.reshape(-1, ).detach().cpu().numpy(), pi_logits.detach().cpu().numpy()
           
        
    def remember(self, state, action, reward, done, log_probs, value_estimates, i):
        self.states[i] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.actions[i] = torch.tensor(action, dtype=torch.int64, device=self.device)
        self.rewards[i] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.dones[i] = torch.tensor(done, dtype=torch.int64, device=self.device)
        self.log_probs[i] = torch.tensor(log_probs, dtype=torch.float32, device=self.device)
        self.value_estimates[i] = torch.tensor(value_estimates, dtype=torch.float32, device=self.device)
        
        
    def _make_policy_dataset(self, next_state):
        with torch.no_grad():
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            _, _, next_value, _ = self.model(next_state)
            next_value = next_value.reshape(-1, )
            bootstrapped_advantage = 0
            self.advantages = torch.zeros_like(self.rewards).to(self.device)
            
            for i in reversed(range(self.num_steps)):
                next_values = next_value if i == self.num_steps - 1 else self.value_estimates[i + 1]
                deltas = self.rewards[i] + self.gamma * (1 - self.dones[i]) * next_values - self.value_estimates[i]
                self.advantages[i] = bootstrapped_advantage = deltas + self.gamma * self.lambd * (1 - self.dones[i]) * bootstrapped_advantage
                
            self.value_targets = self.advantages + self.value_estimates
            assert self.value_targets.shape == self.value_estimates.shape == self.advantages.shape
            
        # start building the auxiliary phase buffer
        start, end = self.num_envs * self.n_pi_index, self.num_envs * (self.n_pi_index + 1)
        self.auxiliary_states[:, start:end] = self.states.detach().cpu().numpy().copy()
        self.auxiliary_value_targets[:, start:end] = self.value_targets.detach().cpu().numpy().copy()
        self.n_pi_index = (self.n_pi_index + 1) % self.n_pi
        
        # is there a way to check if all of the buffer has been filled correctly? (since the training loop is just 2 nested for loops, feels kinda unreliable)
            
    def policy_phase(self, next_state):
        self._make_policy_dataset(next_state)
        
        states = self.states.flatten(start_dim=0, end_dim=1)
        actions = self.actions.flatten(start_dim=0, end_dim=1)
        old_log_probs = self.log_probs.flatten(start_dim=0, end_dim=1)
        old_value_estimates = self.value_estimates.flatten(start_dim=0, end_dim=1)
        advantages = self.advantages.flatten(start_dim=0, end_dim=1)
        value_targets = self.value_targets.flatten(start_dim=0, end_dim=1)
        
        if self.batch_norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        indices = np.arange(self.num_envs * self.num_steps)
        for i in range(self.num_epochs):
            np.random.shuffle(indices)
            minibatch_indices = np.split(indices, self.num_batches)
            for minibatch_idx in minibatch_indices:
                s = states[minibatch_idx]
                a = actions[minibatch_idx]
                olp = old_log_probs[minibatch_idx]
                ove = old_value_estimates[minibatch_idx]
                adv = advantages[minibatch_idx]
                vt = value_targets[minibatch_idx]
                
                pi, _, value, _ = self.model(s)
                value = value.reshape(-1, )
                log_probs = pi.log_prob(a)
                entropies = pi.entropy()
                
                loss = self._compute_loss(log_probs, olp, value, ove, vt, adv, entropies)
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                self.num_updates += 1
                
                
    def auxiliary_phase(self):
        # self.auxiliary_states [num_steps, num_envs * n_pi, 64, 64, 3]
        # self.auxiliary_pi [num_steps, num_envs * n_pi, action_space]
        self._make_auxiliary_dataset()
        
        for i in range(self.e_aux):
            for j in range(0, self.num_envs * self.n_pi, self.num_aux_rollouts):
                start, end = j, j + self.num_aux_rollouts
                
                auxiliary_states = torch.tensor(self.auxiliary_states[:, start:end], device=self.device, dtype=torch.float32)
                auxiliary_value_targets = torch.tensor(self.auxiliary_value_targets[:, start:end], device=self.device, dtype=torch.float32)
                auxiliary_pi_logits = torch.tensor(self.auxiliary_pi[:, start:end], device=self.device, dtype=torch.float32)
                auxiliary_pi_logits = auxiliary_pi_logits.flatten(start_dim=0, end_dim=1)
                auxiliary_pi = torch.distributions.Categorical(logits=auxiliary_pi_logits)
                
                auxiliary_states = auxiliary_states.flatten(start_dim=0, end_dim=1)
                auxiliary_value_targets = auxiliary_value_targets.flatten(start_dim=0, end_dim=1)
                
                pi, _, value_estimates, auxiliary_value_estimates = self.model(auxiliary_states)
                
                # flatten so no broadcasting warning
                value_estimates = value_estimates.flatten(start_dim=0, end_dim=1)
                auxiliary_value_estimates = auxiliary_value_estimates.flatten(start_dim=0, end_dim=1)
                
                L_aux = 0.5 * F.mse_loss(auxiliary_value_estimates, auxiliary_value_targets)
                pi_loss = torch.distributions.kl_divergence(pi, auxiliary_pi).mean()
                joint_loss = L_aux + self.beta_clone * pi_loss
                
                L_V = 0.5 * F.mse_loss(value_estimates, auxiliary_value_targets)
                
                loss = joint_loss + L_V
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
        
    def _make_auxiliary_dataset(self):
        # self.auxiliary_states [num_steps, num_envs * n_pi, 64, 64, 3]
        with torch.no_grad():
            for i, state in enumerate(self.auxiliary_states.transpose(1, 0, 2, 3, 4)):
                state = torch.tensor(state, device=self.device, dtype=torch.float32)
                _, pi_logits, _, _ = self.model(state) # pi logits, need to convert to Categorical since cant store this object in a tensor
                self.auxiliary_pi[:, i] = pi_logits.detach().cpu().numpy().copy() 
        
                
    def _compute_loss(self, log_probs, old_log_probs, value, old_value_estimates, value_targets, advantages, entropies):
        probability_ratio = torch.exp(log_probs - old_log_probs)
        l_clip = torch.mean(
            torch.min(
                advantages * probability_ratio,
                torch.clamp(probability_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            )
        )
        if self.vf_clip:
            vf_unclipped = (value_targets - value) ** 2
            vf_clipped = old_value_estimates + torch.clamp(value - old_value_estimates, -self.vf_clip, self.vf_clip)
            vf_clipped = (value_targets - vf_clipped) ** 2
            l_vf = torch.mean(torch.max(vf_unclipped, vf_clipped))
        else:
            l_vf = (value_targets - value) ** 2
            l_vf = torch.mean(l_vf)
        
        s = torch.mean(entropies)
        
        loss = -l_clip + self.c1 * l_vf - self.c2 * s
        return loss