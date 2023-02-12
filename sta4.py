import random
import numpy as np
from collections import deque


class STA: # revamped for vectorized envs
    def __init__(self):
        '''
        self.vectorized_checkpoints: list of num_envs lists
            * each inner list contains checkpoints (qualified states for STA) for that respective environment 
            * each checkpoint is a list of format [cdata, state (np.array), future_return, old_return]
                
        For self.vectorized_checkpoints
            * the first index should access the corresponding env 
            * the second index corresponds to the checkpoint
            
        CHANGES TO MAKE:
            * change best threshold to 1 number
                * just add the average returns whenever any environment ends instead of calling it after 256 time steps (for PPO), makes it easier to track
                * prob just add an extra function for this
            * change set to 1 list
        '''  
        # new update_set parameters
        self.num_envs = num_envs
        self.sample_num = sample_num
        self.future_steps_num = future_steps_num
        self.best_threshold = 2
        self.average_returns = [0] * 40
        self.average_returns_current_idx = 0
        self.checkpoints = []
        
    def update_set(self, trajectory, old_id):
        '''
        Randomly sample a set of states a trajectory generated during training and add valid states to the STA set
        
        valid states meet the following criteria:
            * controllable; i.e the next 50 steps have a reward greater than some threshold
        
        !!! For ProcGen and PPO, trajectories (vectorized or unvectorized) are littered with random dones throughout the trajectory since there is no manual resets or an efficienct way to sync 
        all of the envs together. Need to account for this when defining controllable states and running the is_qualified checks
            i.e a trajectory (along one env) might be [s_10, s_11, s_12, terminated, s_0, s_1, ...]
            
            If we store time step rewards and accumulate them in this function, we can't trace back states that started in a previous trajectory list
            Instead we can store accumulated rewards which should work
            
            i.e each trajectory entry will be [vectorized_cdata, vectorized_state, vectorized_accumulated_reward, vectorized_done, vectorized_episode_length] 
            where the dones signify a terminal state
        '''
        assert len(old_ids) == self.num_envs
        
        self.average_returns.sort()
        threshold = self.average_returns[20] # middle
        if threshold > self.best_threshold:
            self.best_threshold = threshold
        
        usable_states = [[] for i in range(self.num_envs)]
        vectorized_dones = [i[3] for in trajectory]
        single_env_flattened_dones = [np.array(vectorized_dones)[:, i] for i in range(self.num_envs)] # [[env_1_dones], [env_2_dones],..., [env_n_dones]] where env_n_dones should be [256, ]
        
        for index, flattened_done in enumerate(single_env_flattened_dones):
            for timestep in range(0, len(trajectory) - self.future_steps_num): # i think we can hardcode len(trajectory) to 256
                termination = (flattened_done[timestep:timestep + self.future_steps_num] == 1).any() # checking if the trajectory for this single env terminates within the next 50 steps
                if not termination: 
                    usable_states[index].append(timestep)
        
        old_returns = np.zeros((self.num_envs, ))
        for env_index, old_id in enumerate(old_ids):
            # make sure the starting state of this trajectory is still controllable, i.e replay of this state doesn't terminate after 50 steps
            if old_id >= 0 and usable_states[env_index][0] == 0: 
                old_return = self.checkpoints[old_id][3]
                new_future_return = trajectory[50][2][env_index] 
                if self.checkpoints[old_id][2] < new_future_return:
                    self.checkpoints[old_id][2] = new_future_return
                old_returns[env_index] = old_return
        
        # usable timesteps (controllable states) for each individual environment in the vectorized batch
        sampled_usable_indices = [random.sample(i, 5) for i in usable_states]
        for env_index, indices in enumerate(sampled_usable_indices): # list of indices
            for timestep in indices: # iterating through single index at a time
                future_return = trajectory[timestep + 50][2][env_index] - trajectory[timestep][2][env_index]
                average_future_return = future_return / self.future_steps_num
                
                if average_future_return > threshold and average_future_return > self.best_threshold:
                    checkpoint = [
                            trajectories[timestep][0][env_index], # single cdata entry - might need to turn back into a list later
                            trajectories[timestep][1][env_index], # single state of shape [observation_shape]
                            future_return,
                            old_returns[env_index]
                        ]
                    if len(self.checkpoints) < 20:
                        self.checkpoints.append(checkpoint)
                    else:
                        checkpoint_indices = random.sample(list(range(len(self.checkpoints))), 10)
                        lowest_checkpoint_index = checkpoint_indices.pop(0)
                        lowest_future_return = self.checkpoints[lowest_checkpoint_index][2]
                        
                        for checkpoint_index in checkpoint_indices:
                            if self.checkpoints[checkpoint_index][2] < lowest_future_return:
                                lowest_future_return = self.checkpoints[checkpoint_index][2]
                                lowest_checkpoint_index = checkpoint_index
                        
                        if self.checkpoints[lowest_checkpoint_index][2] >= self.best_threshold:
                            self.checkpoints.append(checkpoint)
                        else:
                            self.checkpoints[lowest_checkpoint_index] = checkpoint
    
    def update_average_returns(self, episode_returns, episode_lengths):
        # episode_returns and episode_lengths might be arrays rather than scalars
        
        for episode_return, episode_length in zip(episode_returns, episode_lengths):
            average_return = episode_return / episode_length
            if average_return > 1:
                self.average_returns[self.average_returns_current_idx] = average_return
                self.average_returns_current_idx += 1
                self.average_returns_current_idx = self.average_returns_current_idx % len(self.average_returns)
        
        
    def select_state_indices(self, agent):
        '''
        Randomly sample sample_num checkpoints. If any of these checkpoints has an average future return greater than the current
        highest average future return, return it as the initial state for the next trajectory.
        
        If more than one of the sampled checkpoints have average future returns greater than the current highest average future 
        return, select the one that the agent is most unfamiliar with (lowest Q-value)
        
        In the vectorized case, selecting a valid checkpoint is not as straightforward as the unvectorized case
        When checking the qualified states in a checkpoint (there may be multiple since each checkpoint contains a batch of envs), 
        some of the envs may be valid vandidates while others may not. This makes it difficult to return an entire checkpoint at a 
        time if the condition is for all qualified states to be a valid candidate.
        
        Current workaround:
            Iterate through the Nc sampled checkpoints as usual, but bundle all the valid candidates among the is_qualified states
            as a new checkpoint. If there is more than one valid candidate in the same env, then apply the Q-value evaluation to choose
            as usual in the unvectorized case.
        '''
        valid_candidates, q_values = [-1 for i in range(self.num_envs)], [-1 for i in range(self.num_envs)]
        
        for i in range(self.num_samples):
            for i in range(self.num_envs):
                random_idx = random.randint(0, len(self.checkpoints) - 1)
                avg_future_return = self.checkpoints[random_idx][2] / self.future_num_steps
                if avg_future_return > self.best_threshold:
                    with torch.no_grad():
                        state = self.checkpoints[random_idx][1]
                        _, score = agent.model(state)
                    if all(element != -1 for element in valid_candidates): # all valid candidates spots have been filled and now need to filter using the q value
                        replacement_idx = np.argmax(q_values)
                        if score < q_values[replacement_idx]:
                            q_values[replacement_idx] = score
                            valid_candidates[replacement_idx] = random_idx
                    else:
                        valid_candidates.append(random_idx)
                        q_values.append(score)
        
        assert len(valid_candidates) == len(q_values) == self.num_envs
        return valid_candidates