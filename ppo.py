import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# only supports discrete action spaces for now, update in the future
class PPO:
    '''
    J(Θ) = E[L_CLIP - L_VF + S(π)] gradient ascent, descent method would be negative of this equation
    L_CLIP = E[min(rtΘ * A_t, clip(rtΘ * A_t, 1 - ε, 1 +  ε))]
    L_VF = (V_Θ(s) - V_old(s))^2
    S(π) = entropy 
    '''
    def __init__(self, observation_space, action_space, **params):
        self.observation_space = observation_space.shape
        self.action_space = action_space
        
        self.epsilon = params['epsilon']
        self.lr = params['lr']
        self.gamma = params['gamma']
        self.device = params['device']
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.num_batches = params['num_batches']
        self.c1 = params['c1']
        self.c2 = params['c2']
        
        # changing here but make a parameter in the future
        self.policy = ImpalaCNN(observation_space, action_space).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        
        self.memory = []
        self.reward_buffer = []
        self.done_buffer = []
        
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pi, value = self.policy_old(state)
            action = pi.sample()
            log_probs = pi.log_prob(action)
        
        self.memory.append([state, action, log_probs, value])
        return action.detach().cpu().numpy()
    
    def remember(self, reward, done):
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        
    def _make_dataset(self, batch_size):
        assert len(self.memory) == len(self.reward_buffer) == len(self.done_buffer)
        
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.reward_buffer), reversed(self.done_buffer)):
            discounted_reward = reward + self.gamma * discounted_reward 
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7).view(-1, ) # [1000, 16]
        rewards = rewards.view(-1, )
        
        # need to make sure old_log_probs, log_probs, and advantages are the same shape
        states = torch.stack([i[0] for i in self.memory], dim=0).to(self.device).view(-1, *self.observation_space) # changed this for image format
        actions = torch.stack([i[1] for i in self.memory], dim=0).to(self.device).view(-1, )
        old_log_probs = torch.stack([i[2] for i in self.memory], dim=0).to(self.device).view(-1, )
        old_value_estimates = torch.stack([i[3] for i in self.memory], dim=0).to(self.device).view(-1, )
        
        advantage = rewards.detach() - old_value_estimates.detach()
        
        self.states = states
        self.actions = actions
        self.old_log_probs = old_log_probs
        self.old_value_estimates = old_value_estimates
        self.advantage = advantage
        
    def _make_batches(self):
        minibatches = []
        for i in range(self.num_batches):
            random_idx = np.random.randint(low=0, high=len(self.memory), size=self.batch_size)
            s = self.states[random_idx]
            a = self.actions[random_idx]
            olp = self.old_log_probs[random_idx]
            ove = self.old_value_estimates[random_idx]
            av = self.advantage[random_idx]
            minibatches.append([s, a, olp, ove, av])
        return minibatches
    
    def _compute_loss(self, pi, value, old_value_estimates, probability_ratio, advantage):
        L_CLIP = torch.min(
                            probability_ratio * advantage,
                            torch.clamp(probability_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
                    )
        L_VF = F.mse_loss(value, old_value_estimates)
        S = pi.entropy()
        
        loss = -L_CLIP + self.c1 * L_VF - self.c2 * S
        loss = torch.mean(loss)
        return loss
    
    def learn(self):
        '''
        for e in range(epochs):
            minibatches <- get_minibatch()
            for minibatch in minibatches:
                update()
        
        minibatches <- get_minibatch()
        for e in range(epochs):
            update()
            
        i.e new set of minibatches for each epoch, or same set of minibatches for all epochs?
        '''
        batch_size = min(self.batch_size, len(self.memory))
        self._make_dataset(batch_size)
        
        for e in range(self.epochs):
            minibatch = self._make_batches()
            for states, actions, old_log_probs, old_value_estimates, advantage in minibatch:
                pi, value = self.policy(states)
                value = value.view(-1, )
                log_probs = pi.log_prob(actions)
                probability_ratio = torch.exp(log_probs - old_log_probs.detach())
                
                loss = self._compute_loss(pi, value, old_value_estimates, probability_ratio, advantage)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        torch.cuda.empty_cache()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory = []
        self.done_buffer = []
        self.reward_buffer = []