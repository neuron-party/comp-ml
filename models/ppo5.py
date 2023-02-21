import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# same thing as PPO4 but with some minor changes, specifically the shape of the storage arrays but should work identically
class PPO5:
    def __init__(self, observation_space, action_space, model, **params):
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.gamma = params['gamma']
        self.lambd = params['lambd']
        self.lr = params['lr']
        self.epsilon = params['epsilon']
        self.vf_clip = params['vf_clip']
        self.max_grad_norm = params['max_grad_norm']
        self.norm_adv = params['norm_adv']
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
        
        self.states = torch.zeros((self.num_steps, self.num_envs, *self.observation_space)).to(self.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs, self.action_space)).to(self.device)
        self.log_probs = torch.zeros((self.num_steps, self.num_envs, self.action_space)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.value_estimates = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        
    def get_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            pi, value = self.model(state)
            action = pi.sample()
            log_probs = pi.log_prob(action)
            
        return action.detach().cpu().numpy(), log_probs.detach().cpu().numpy(), value.reshape(-1, ).detach().cpu().numpy()
            
    def remember(self, state, action, reward, done, log_probs, value_estimates, i):
        self.states[i] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.actions[i] = torch.tensor(action, dtype=torch.int64, device=self.device)
        self.rewards[i] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.dones[i] = torch.tensor(done, dtype=torch.int64, device=self.device)
        self.log_probs[i] = torch.tensor(log_probs, dtype=torch.float32, device=self.device)
        self.value_estimates[i] = torch.tensor(value_estimates, dtype=torch.float32, device=self.device)
        
    def _make_dataset(self, next_state):
        with torch.no_grad():
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            _, next_value = self.model(next_state)
            next_value = next_value.reshape(-1, )
            bootstrapped_advantage = 0
            self.advantages = torch.zeros_like(self.rewards).to(self.device)
            
            for i in reversed(range(self.num_steps)):
                next_values = next_value if i == self.num_steps - 1 else self.value_estimates[i + 1]
                deltas = self.rewards[i] + self.gamma * (1 - self.dones[i]) * next_values - self.value_estimates[i]
                self.advantages[i] = bootstrapped_advantage = deltas + self.gamma * self.lambd * (1 - self.dones[i]) * bootstrapped_advantage
                
            self.value_targets = self.advantages + self.value_estimates
            assert self.value_targets.shape == self.value_estimates.shape == self.advantages.shape
            
    def learn(self, next_state):
        self._make_dataset(next_state)
        
        states = self.states.flatten(start_dim=0, end_dim=1)
        actions = self.actions.flatten(start_dim=0, end_dim=1)
        old_log_probs = self.log_probs.flatten(start_dim=0, end_dim=1)
        old_value_estimates = self.value_estimates.flatten(start_dim=0, end_dim=1)
        advantages = self.advantages.flatten(start_dim=0, end_dim=1)
        value_targets = self.value_targets.flatten(start_dim=0, end_dim=1)
        
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
                
                if self.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                
                pi, value = self.model(s)
                value = value.reshape(-1, )
                log_probs = pi.log_prob(a)
                entropies = pi.entropy()
                
                # import pdb; pdb.set_trace()
                
                loss = self._compute_loss(log_probs, olp, value, ove, vt, adv, entropies)
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                self.optimizer.step()
                
                self.num_updates += 1
                
    def _compute_loss(self, log_probs, old_log_probs, value, old_value_estimates, value_targets, advantages, entropies):
        assert log_probs.shape == old_log_probs.shape == entropies.shape
        if len(log_probs.shape) > 1: # multidimensional action
            advantages = advantages.unsqueeze(-1) # unsqueeze for broadcasting
        
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