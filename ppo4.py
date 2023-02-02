class PPO4:
    def __init__(self, model, **params):
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.gamma = params['gamma']
        self.lr = params['lr']
        self.epsilon = params['epsilon']
        self.vf_clip = params['vf_clip']
        
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
        self.actions = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.log_probs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.value_estimates = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        
    def get_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            pi, value = self.model(state)
            action = pi.sample()
            log_probs = pi.log_prob(action)
            
        return action.detach().cpu().numpy(), log_probs.detach().cpu().numpy(), value.detach().cpu().numpy()
            
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
            bootstrapped_advantage = 0
            self.advantages = torch.zeros_like(self.rewards).to(self.device)
            
            for i in reversed(range(self.num_steps)):
                next_values = next_value if i == self.num_steps - 1 else self.value_estimates[i + 1]
                deltas = self.rewards[i] + self.gamma * self.dones[i] * next_values - self.value_estimates[i]
                self.advantages[i] = bootstrapped_advantage = deltas + self.gamma * self.lambd * self.dones[i] * bootstrapped_advantage
                
            self.value_targets = self.advantages + self.value_estimates
            assert self.value_targets.shape == self.value_estimates.shape == self.advantages.shape
            
    def learn(self):
        states = self.states.flatten(start_dim=0, end_dim=1)
        actions = self.actions.flatten(start_dim=0, end_dim=1)
        old_log_probs = self.log_probs.flatten(start_dim=0, end_dim=1)
        old_value_estimates = self.value_estimates.flatten(start_dim=0, end_dim=1)
        advantages = self.advantages.flatten(start_dim=0, end_dim=1)
        value_targets = self.value_targets.flatten(start_dim=0, end_dim=1)
        
        indices = np.arange(self.num_envs * self.num_steps)
        for i in range(self.num_epochs):
            np.random.shuffle(indices)
            minibatch_indices = np.split(indices, self.num_batches):
            for minibatch_idx in minibatch_indices:
                s = states[minibatch_idx]
                a = actions[minibatch_idx]
                olp = old_log_probs[minibatch_idx]
                ove = old_value_estimates[minibatch_idx]
                adv = advantages[minibatch_idx]
                vt = value_targets[minibatch_idx]
                
                pi, value = self.model(s)
                log_probs = pi.log_prob(a)
                entropies = pi.entropy()
                
                loss = self._compute_loss(log_probs, olp, value, ove, vt, adv, entropies)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                self.num_updates += 1
                
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