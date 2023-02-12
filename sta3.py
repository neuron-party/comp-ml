import random
import numpy as np


class STA: # revamped for vectorized envs
    def __init__(self):
        '''
        self.vectorized_checkpoints: list of num_envs lists
            * each inner list contains checkpoints (qualified states for STA) for that respective environment 
            * each checkpoint is a list of format [cdata, state (np.array), future_return, old_return]
                
        For self.vectorized_checkpoints
            * the first index should access the corresponding env 
            * the second index corresponds to the checkpoint
        '''
        # update_set parameters
        self.num_envs = num_envs
        self.sample_num = sample_num
        self.future_steps_num = future_steps_num
        self.best_threshold = np.ones((num_envs, )) * 2 # [num_envs, ]
        self.average_returns = np.ones((num_envs, 40)) # [num_envs, 40]
        self.average_returns_current_idx = np.zeros((num_envs, )) # [num_envs, ]
        self.vectorized_checkpoints = [[] for i in range(self.num_envs)] # [[],...,[]] num_envs times
        
        # sample_set parameters
        self.nc = nc # 3
        
        
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
        
        for row in self.average_returns:
            row.sort()

        threshold = self.average_returns[:, 20] # [num_envs, ]
        self.best_threshold[threshold > self.best_threshold] = threshold[threshold > self.best_threshold]
        
        # i dont think this check is sufficient
        # i think we can change it to an array of shape [num_envs, ], where each element is the number of usable states for each single env, then sample from this array (nevermind)
        # also, usable states are ones where there are at least 50 + 12 time steps in front of it since thats the definition we are using for controllable
        # i don't think we can just leave each as a number since they are all out of order, we need to literally define every single time-step that is usable for each env
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
            if old_id >= 0 and usable_states[env_index][0] == 0: # make sure the starting state of this trajectory is still controllable, i.e replay of this state doesn't terminate after 50 steps
                old_return = self.checkpoints[env_index][old_id][3]
                new_future_return = trajectory[50][2][env_index] 
                if self.checkpoints[env_index][old_id][2] < new_future_return:
                    self.checkpoints[env_index][old_id][2] = new_future_return
                old_returns[env_index] = old_return
        
        
        sampled_usable_indices = [random.sample(i, 5) for i in usable_states] # random indices for each env in the vectorized envs; ill change the sample number later cuz i need to write some logic for timesteps
        for env_index, indices in enumerate(sampled_usable_indices): # list of indices
            for timestep in indices: # iterating through single index at a time
                future_return = trajectory[timestep + 50][2][env_index] - trajectory[timestep][2][env_index]
                average_future_return = future_return / self.future_steps_num
                
                if average_future_return > threshold[env_index] and average_future_return > self.best_threshold[env_index]:
                    checkpoint = [
                            trajectories[timestep][0][env_index], # single cdata entry - might need to turn back into a list later
                            trajectories[timestep][1][env_index], # single state of shape [observation_shape]
                            future_return,
                            old_returns[env_index]
                        ]
                            
                    if len(self.vectorized_checkpoints[env_index]) < 20:
                        self.vectorized_checkpoints[env_index].append(checkpoint)
                    else:
                        checkpoint_indices = random.sample(list (range(len(self.vectorized_checkpoints[env_index]))), 10)
                        lowest_checkpoint_index = checkpoint_indices.pop(0)
                        lowest_future_return = self.vectorized_checkpoints[env_index][lowest_checkpoint_index][2]

                        for checkpoint_index in checkpoint_indices:
                            if self.vectorized_checkpoints[env_index][checkpoint_index][2] < lowest_future_return:
                                lowest_future_return = self.vectorized_checkpoints[env_index][checkpoint_index][2]
                                lowest_checkpoint_index = checkpoint_index

                        if self.vectorized_checkpoints[env_index][lowest_checkpoint_index][2] >= self.best_threshold[env_index]:
                            self.vectorized_checkpoints[env_index].append(checkpoint)
                        else:
                            self.vectorized_checkpoints[env_index] = checkpoint
                                    
        # total_return and average_return are used for setting the threshold
        # accumulated_rewards (and thus future_return and average_future_return) are whats used for the is_qualified check; these don't require a complete episode
        vectorized_accumulated_rewards = [i[2] for i in trajectory]
        vectorized_dones = [i[3] for i in trajectory]
        vectorized_episode_lengths = [i[4] for i in trajectory]
        
        # there is also the case where the same env has multiple terminations within the same 256 num_steps trajectory, so i think we need to do the update inside the loop
        for var, vd, vel in zip(vectorized_accumulated_rewards, vectorized_dones, vectorized_episode_lengths):
            # var, vd, vel are each of shape [num_envs, ]
            if vd.any(): # i think numpy throws an error if you try to iterate through an empty array
                env_indices = np.where(vd == 1)[0]
                for index in env_indices:
                    single_env_average_return = var[index] / vel[index]
                    if single_env_average_return > 1:
                        average_returns_current_idx = self.average_returns_current_idx[index]
                        self.average_returns[index][average_returns_current_idx] = single_env_average_return
                        self.average_returns_current_idx[index] += 1
                        self.average_returns_current_idx[index] = self.average_returns_current_idx[index] % len(self.average_returns[index])
        
        
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
        valid_candidates = np.ones((self.num_envs, )) * -1
        q_value_matrix = np.empty((self.num_envs, ))
        
        for i in range(self.nc):
            vectorized_candidate_indices = np.ones((self.num_envs, )) * -1
            temp_q_value_matrix = np.zeros((self.num_envs, ))
            
            for env_index in range(self.num_envs):
                random_idx = random.randint(0, len(self.checkpoints[env_index]) - 1)
                with torch.no_grad():
                    state = self.checkpoints[env_index][random_idx][1]
                    _, q_value = agent.model(state)
                avg_future_return = self.checkpoints[env_index][random_idx][2] / self.future_steps_num
                if avg_future_return > self.best_threshold[env_index]:
                    vectorized_candidate_indices[env_index] = random_idx
                    temp_q_value_matrix[env_index] = q_value
                    
            replacement_mask = np.logical_and(valid_candidates != -1, valid_candidate_indices != -1) # if some envs have more than 1 valid candidate, we choose using the lower q value
            if replacement_mask.any():
                replacement_mask_2 = q_value_matrix[replacement_mask] > temp_q_value_matrix[replacement_mask]
                valid_candidates[replacement_mask][replacement_mask_2] = valid_candidate_indices[replacement_mask][replacement_mask_2]
                
            q_value_matrix = temp_q_value_matrix
            
            # at this point we've already counted for all the overlaps, so now we just add the rest of the new indices in
            no_overlap_mask = valid_candidates == -1
            remaining_indices_mask = vectorized_candidate_indices[no_overlap_mask] != -1
            valid_candidates[no_overlap_mask][remaining_indices_mask] == vectorized_candidate_indices[no_overlap_mask][remaining_indices_mask]
            
        # for actually resetting the states, if an index element == -1, just keep the current state dynamics, if != -1, then we change to augment
        return valid_candidates