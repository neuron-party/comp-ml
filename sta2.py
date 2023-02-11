class STA: # revamped for vectorized envs
    def __init__(self):
        '''
        self.checkpoints: 
            each entry is a dictionary with cdata, state, and future_ret
                state and rewards are arrays of shape [num_envs, observation_space] and [num_envs, 1] respectively
                
        For self.vectorized_checkpoints
            * the first index should access the corresponding env 
            * the second index corresponds to the checkpoint
        '''
        self.num_envs = num_envs
        self.sample_num = sample_num
        self.future_steps_num = future_steps_num
        self.best_threshold = np.ones((num_envs, 1)) * 2 # [num_envs, 1]
        self.average_returns = np.ones((num_envs, 40)) # [num_envs, 40]
        self.vectorized_checkpoints = [[] for i in range(self.num_envs)] # [[],...,[]] num_envs times
        
        
    def update_set(self, trajectory, old_id):
        '''
        Randomly sample a set of states from a trajectory generated during training and add valid states to the STA set
        
        valid states meet the following criteria:
            * controllable; i.e the next 50 steps have a reward greater than some threshold
        
        trajectory: list of lists where each entry is 
        [state_cdata, state, reward]
        
        wait... another check I need to do for procgen vectorized environments is when checking for controllable states,
        some trajectories might have random dones in the middle of them and i need to slice them accordingly. i will add this tomorrow
        '''
        
        # i need to check for dones and properly define controllable states here since procgen trajectories are littered with random dones
        # and are all out of sync
        
        rewards = [i[2] for i in trajectory] # [[num_envs, ],...,[num_envs, ]] 256 times (for PPO)
        accumulated_rewards = [np.zeros((self.num_envs, ))] + list(itertools.accumulate(rewards)) # [[num_envs, ],...,[num_envs, ]] 257 times
        accumulated_rewards = np.array(accumulated_rewards) # [256, num_envs] 
        total_return = accumulated_rewards[-1] # [num_envs, ]
        
        # !!! figure out a way to update the future_ret with old_id and stuff
        
        average_return = total_return / len(rewards) # [num_envs, ]
        for row in self.average_returns:
            row.sort()

        threshold = self.average_returns[:, 20] # [num_envs, ]
        self.best_threshold[threshold > self.best_threshold] = threshold[threshold > self.best_threshold]
        
        # i dont think this check is sufficient
        # i think i need to 
        usable_states = len(trajectory) - self.future_steps_num - 12 
        if usable_states > 0:
            indices = random.sample(
                list(range(10, len(trajectory) - self.future_steps_num - 1)),
                int(len(trajectory) / self.future_steps_num)
            )
            count = 0
            for idx in indices:
                future_return = accumulated_rewards[idx + self.future_steps_num + 1, :] - accumulated_rewards[idx + 1, :] # [num_envs, ]
                average_future_return = future_return / self.future_steps_num # [num_envs, ]
                
                is_qualified_mask = np.logical_and(
                    average_future_return > threshold,
                    average_future_return > self.best_threshold
                ) # [num_envs, ]
                
                assert len(is_qualified_mask) == len(self.vectorized_checkpoints)
                
                if is_qualified_mask.any(): # if there are any envs with qualified states, we proceed with the loop
                    for index, qualified in enumerate(is_qualified_mask):
                        # looping through each env in the vectorized env, and updating its respective checkpoint list if there are
                        # qualified states
                        
                        # trajectories[idx]: list of [vectorized_cdata, vectorized_states, vectorized_rewards]
                        # trajectories[idx][0]: vectorized_cdata with length num_envs
                        # trajectories[idx][1]: vectorized_states with shape [num_envs, observation_shape]
                        # trajectories[idx][2]: vectorized_rewards with shape [num_envs, ]
                        if qualified:
                            checkpoint = [
                                    trajectories[idx][0][index], # single cdata entry - might need to turn back into a list later
                                    trajectories[idx][1][index], # single state of shape [observation_shape]
                                    future_return[index]
                                ]
                            
                            if len(self.vectorized_checkpoints[index]) < 20:
                                self.vectorized_checkpoints[index].append(checkpoint)
                            else:
                                checkpoint_indices = random.sample(list(range(len(self.vectorized_checkpoints[index]))), 10)
                                lowest_checkpoint_index = checkpoint_indices.pop(0)
                                lowest_future_return = self.vectorized_checkpoints[index][lowest_checkpoint_index][2]
                                
                                for checkpoint_index in checkpoint_indices:
                                    if self.vectorized_checkpoints[index][checkpoint_index][2] < lowest_future_return:
                                        lowest_future_return = self.vectorized_checkpoints[index][checkpoint_index][2]
                                        lowest_checkpoint_index = checkpoint_index
                                
                                if self.vectorized_checkpoints[index][lowest_checkpoint_index][2] >= self.best_threshold[index]:
                                    self.vectorized_checkpoints[index].append(checkpoint)
                                else:
                                    self.vectorized_checkpoints[index] = checkpoint
                                                       
        if average_return > 1:
            self.average_returns[self.average_returns_current_idx] = average_return
            self.average_returns_current_idx += 1
            self.average_returns_current_idx = self.average_returns_current_idx % len(self.average_returns)
        
        
    def select_state(self):
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
            
            ! However, this creates an entirely new checkpoint and makes the logic for the old_id and updating future returns much more 
            complicated
        '''
        valid_candidates = []
        
        for i in range(self.sample_num):
            random_idx = random.randint(0, len(self.checkpoints) - 1)
            score = agent.model(self.checkpoints[random_idx]['state'])
            mask = self.checkpoints[random_idx]['is_qualified_mask']
            avg_future_ret = self.checkpoints[random_idx]['future_ret'][mask] / self.future_steps_num
            
            if avg_future_ret > best_threshold:
                valid_candidates.append([self.checkpoints[random_idx], score])
        
        if valid_candidates: # select the one with the lowest score
            return sorted(valid_candidates, key=lambda x: x[1])[0][0]
        
        return None # no valid candidates