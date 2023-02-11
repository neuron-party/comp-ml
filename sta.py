class STA: # revamped for vectorized envs
    def __init__(self):
        '''
        self.checkpoints: 
            each entry is a dictionary with cdata, state, and future_ret
                state and rewards are arrays of shape [num_envs, observation_space] and [num_envs, 1] respectively
        '''
        
        self.num_envs = num_envs
        self.checkpoints = []
        self.sample_num = sample_num # Nc
        self.future_steps_num = future_steps_num # 50
        self.best_threshold = np.ones((num_envs, 1)) * 2 # [all 2s]
        self.average_returns = np.ones((num_envs, 40))
        # self.average_returns = [1] * 40 # use the top 40 returns to compute the threshold
        
    
        
    def update_set(self, trajectory, old_id):
        '''
        Randomly sample a set of states from a trajectory generated during training and add valid states to the STA set
        
        valid states meet the following criteria:
            * controllable; i.e the next 50 steps have a reward greater than some threshold
        
        trajectory: list of lists where each entry is 
        [state_cdata, state, reward]
        '''
        rewards = [i[2] for i in trajectory]
        # [trajectory_length, num_envs]
        accumulated_rewards = np.array([np.zeros((self.num_envs, ))] + list(itertools.accumulate(rewards))) 
        total_return = accumulated_rewards[-1] # [1, num_envs]
        
        # replacing the future reward of a previously encountered state if the future reward of this state is now higher
        # need to fix this, everything else is fixed for vectorized envs
        if old_id and len(trajectory) >= self.future_steps_num:
            mask = self.checkpoints[old_id]['is_qualified_mask']
            # replacing the env indices where the future return is now higher
            replacement_mask = self.checkpoints[old_id]['future_ret'][mask] < accumulated_rewards.transpose(1, 0)[self.future_steps_num]
            self.checkpoints[old_id]['future_ret'][mask][replacement_mask] = accumulated_rewards.transpose(1, 0)[self.future_steps_num][mask][replacement_mask]
                      
#             if self.checkpoints[old_id]['future_ret'] < accumulated_rewards[self.future_steps_num]:
#                 self.checkpoints[old_id]['future_ret'] = accumulated_rewards[self.future_steps_num]
        
        average_return = total_return / len(rewards) # [1, num_envs]
        for row in self.average_returns:
            row.sort()

        # the index below is just the middle of the list if train_trajs_top_ratio = 0.5
        # threshold = self.average_returns[int(len(self.average_returns) * (1 - train_trajs_top_ratio))] 
        # i have the threshold index hardcoded right now, can change in future if needed
        threshold = self.average_returns[:, 20] # [num_envs, 1]
        # updating the best threshold
        self.best_threshold[threshold > self.best_threshold] = threshold[threshold > self.best_threshold]
        
        usable_states = len(trajectory) - self.future_steps_num - 12 # why -12 here?
        if usable_states > 0:
            # only get states from the 10th state forward up to the last state (exclude the first 10 states; why?)
            # if the trajectory is 100 states, then sample 2
            # if trajectory is 1000 states, then sample 20
            indices = random.sample(
                list(range(10, len(trajectory) - self.future_steps_num - 1)),
                int(len(trajectory) / self.future_steps_num)
            )
            count = 0
            for idx in indices:
                # why add 1 to both indices here? why not just idx + self.f_s_n and idx?
               
                future_return = accumulated_rewards[idx + self.future_steps_num + 1, :] - accumulated_rewards[idx + 1, :] # [1, num_envs]
                future_return = future_return.transpose(1, 0) # [num_envs, 1]
                average_future_return = future_return / self.future_steps_num # [num_envs, 1]
                
                
                is_qualified_mask = np.logical_and(
                    average_future_return > threshold,
                    average_future_return > self.best_threshold
                ) # [num_envs, 1]
                
                if is_qualified_mask.any(): # if there is a single state of the vectorized states that is qualified, we proceed with
                # the loop
                
                # set of states that qualify to be added into the set
                # cdata is initially just a normal list so need to convert to array to do logical indexing
                # cdata = trajectory[idx][0][is_qualified_mask]  
                # state = trajectory[idx][1][is_qualified_mask]
                
                # when loading a checkpoint, apply the mask to the vectorized environments so that we only reset the specific envs
                # that are qualified, don't reload all envs.
                # we use to the mask to properly change the cdata list, replacing specific elements with cdata elements where we need
                    checkpoint = {
                        'cdata': trajectory[idx][0], # [num_envs, 1]
                        'state': trajectory[idx][1], # [num_envs, observation_shape]
                        'future_ret': future_return, # [num_envs, 1]
                        'is_qualified_mask': is_qualified_mask # [num_envs, 1] boolean mask
                    }
                    
                    # when setting the state, just apply the is_qualified_mask after calling a env.get_state(), so that only the 
                    # envs we want to augment are changed while leaving everything else intact
                    
                    if len(self.checkpoints) < 20:
                        self.checkpoints.append(checkpoint)
                    else: # randomly sampling 10 states currently in the set; if any of these sets has a lower future return than
                        # the current best threshold, then replace that state instead of simply appending to remove some of the 
                        # worse states and the agent gets better
                        
                        # this method of replacing the lowest return state is not as straightforward anymore since all checkpoints
                        # may have a different set of environments with qualified states
                        
                        # idea for vectorized case:
                            # out of the 10 checkpoints sampled, replace the checkpoint with the lowest better-than-ratio
                            # example: 2 of the future returns > best threshold and 8 of the future returns < best threshold 
                            # (so 10 qualified envs total). the ratio for this checkpoint would be 0.2
                            # if the lowest ratio is > 0.5, then add it as a checkpoint and dont replace anything
                            
                            # however I still don't think this solution is ideal
                        checkpoint_indices = random.sample(list(range(len(self.checkpoints))), 10)
                        lowest_ratio = 1.0
                        lowest_checkpoint_idx = 0
                        
                        for checkpoint_idx in checkpoint_indices:
                            mask = self.checkpoints[checkpoint_idx]['is_qualified_mask']
                            qualified_future_returns = self.checkpoints[checkpoint_idx]['future_ret'][mask]
                            qualified_best_threshold = self.best_threshold[mask]
                            
                            assert len(qualified_future_returns) == len(qualified_best_thresold)
                            
                            ratio = sum(qualified_future_returns >= qualified_best_threshold) / len(qualified_future_returns)
                            if ratio < lowest_ratio:
                                lowest_ratio = ratio
                                lowest_checkpoint_idx = checkpoint_idx
                        
                        if lowest_ratio < 0.5:
                            self.checkpoints[lowest_checkpoint_idx] = checkpoint
                        else:
                            self.checkpoints.append(checkpoint)
                            
                            
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