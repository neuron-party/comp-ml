# more memory efficient sta for continuous environments like starpilot
import os
import gym
import wandb
import procgen
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

from models.ppo import *
from models.policies import *
from NEW_STA import *


def parse_args():
    parser = argparse.ArgumentParser()
    
    # agent args
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--vf-clip', type=float, default=0.2)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num-steps', type=int, default=256)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--num-batches', type=int, default=8)
    parser.add_argument('--lambd', type=float, default=0.95)
    parser.add_argument('--c1', type=float, default=0.5)
    parser.add_argument('--c2', type=float, default=0.01)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--norm-adv', type=bool, default=True)
    
    # env args
    parser.add_argument('--env-name', type=str, default='jumper')
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--num-levels', type=int, default=500)
    parser.add_argument('--start-level', type=int, default=0)
    parser.add_argument('--distribution-mode', type=str, default='hard')
    parser.add_argument('--max-global-steps', type=int, default=200000000)
    
    # saving/logging args
    parser.add_argument('--log', type=bool, default=False)
    parser.add_argument('--project-name', type=str, default='project')
    parser.add_argument('--instance-name', type=str, default='trial1')
    parser.add_argument('--save-path', type=str, default='agent.pth')
    
    # sta args
    parser.add_argument('--sta-type', type=int, default=0)
    
    args = parser.parse_args()
    return args

def main(args):
    if args.log:
        wandb.init(
            project=args.project_name,
            name=args.instance_name,
            sync_tensorboard=True
        )
        
    writer = SummaryWriter(f'runs/{args.instance_name}')
    
    env = procgen.ProcgenEnv(
        num_envs=args.num_envs, 
        env_name=args.env_name, 
        num_levels=args.num_levels, 
        start_level=args.start_level,
        distribution_mode=args.distribution_mode
    )
    env = gym.wrappers.TransformObservation(env, lambda x: x['rgb'])
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.is_vector_env = True
    env.single_action_space = env.action_space.n
    env.single_observation_space = env.observation_space['rgb']
    
    model = ImpalaCNN(env.single_observation_space, env.single_action_space)
    
    device = torch.device('cuda:' + str(args.device))
    params = {
        'gamma': args.gamma, 'lr': args.lr, 'epsilon': args.epsilon, 'vf_clip': args.vf_clip, 'device': device, 'num_steps': args.num_steps,
        'num_epochs': args.num_epochs, 'num_batches': args.num_batches, 'num_envs': args.num_envs, 'lambd': args.lambd, 
        'c1': args.c1, 'c2': args.c2, 'max_grad_norm': args.max_grad_norm, 'norm_adv': args.norm_adv
    }
    
    agent = PPO4(env.single_observation_space.shape, env.single_action_space, model, **params)
    
    # sta_type 0: upper bound sta
    # sta_type 1: li-cheng's idea of controllable states
    # sta_type 2: combination of the 2, no upper bound
    # sta_type 3: STAC, for envs like starpilot
    if args.sta_type == 0:
        STA_YUH = STA1(num_envs=args.num_envs, num_samples=3, future_steps_num=50, best_threshold=0)
    elif args.sta_type == 1:
        STA_YUH = STA2(num_envs=args.num_envs, num_samples=3, future_steps_num=50, best_threshold=0)
    elif args.sta_type == 2:
        STA_YUH = STA3(num_envs=args.num_envs, num_samples=3, future_steps_num=50, best_threshold=0)
    elif args.sta_type == 3: # for continuous envs like starpilot, experimenting to see if STA works better for continuous envs
        STA_YUH = ProcgenMemoryEfficientSTA(num_envs=args.num_envs, num_samples=3, future_steps_num=50, best_threshold=0)
    
    # training loop
    # training loop
    vectorized_accumulated_rewards = np.zeros((args.num_envs, ))
    vectorized_episode_lengths = np.zeros((args.num_envs, ))
    old_ids = np.ones((args.num_envs, )) * -1
    max_global_steps = args.max_global_steps
    total_global_steps = 0
    state = env.reset()
    
    while total_global_steps < max_global_steps:
        trajectory = []
        for i in range(args.num_steps):
            action, log_probs, value_estimate = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            vectorized_episode_lengths += 1
            vectorized_accumulated_rewards += reward
            
            # need .copy()
            trajectory.append([
                env.env.env.env.get_state(),
                vectorized_accumulated_rewards.copy(),
                done, 
                vectorized_episode_lengths.copy()
            ])

            if done.any(): # on termination of any of the environments, we update the average returns in the STA object
                STA_YUH.update_average_returns(
                    episode_returns=vectorized_accumulated_rewards[done].copy(),
                    episode_lengths=vectorized_episode_lengths[done].copy()
                )

                # avg_rewards.append(np.mean(vectorized_accumulated_rewards[done]))

                vectorized_accumulated_rewards[done] = 0
                vectorized_episode_lengths[done] = 0

            agent.remember(state, action, reward, done, log_probs, value_estimate, i)

            state = next_state
            total_global_steps += args.num_envs
            
            for item in info:
                if 'episode' in item.keys():
                    writer.add_scalar('charts/episodic_return', item['episode']['r'], total_global_steps)

        agent.learn(state)

        STA_YUH.update_set(trajectory, old_ids)
        old_ids = np.ones((args.num_envs, )) * -1

        # if random.random() < ratio:
        if random.random() < 0.5: # li-cheng set new ratio to 0.5
            if len(STA_YUH.checkpoints) > 200: # how do update the vectorized accumulated rewards and vectorized episode lengths accordingly?
                ids = STA_YUH.select_state_indices(agent)
                current_cdata = env.env.env.env.get_state()
                current_cdata_copy = current_cdata.copy()
                
                q_value_list = []
                valid_idxes = []
                
                num_valid_states_sampled = len(ids)
                
                # pseudocode:
                while len(ids) > 0: # min length 0, max length num_envs * num_samples (i.e 196)
                    i = 0
                    while i < args.num_envs:
                        if len(ids) < 1:
                            break
                        idx = ids.pop()
                        current_cdata[i] = STA_YUH.checkpoints[idx][0]
                        i += 1
                        valid_idxes.append(idx)
                        
                    # there are more valid state candidates
                    # if len(ids) > 0:
                    env.env.env.env.set_state(current_cdata)
                    state = env.reset() # manual reset warning but we can ignore this to get the actual states after resetting
                    with torch.no_grad():
                        _, _, q_values = agent.get_action(state) # (64, )
                    for q in q_values[:i]: # need [:i] here since we dont want q values of non-sta sampled states
                        q_value_list.append(q)
                            
                print(f'Q value list length: {len(q_value_list)}, Valid indices list length: {len(valid_idxes)}')
                assert len(q_value_list) == len(valid_idxes)
                
                old_ids = [-1 for i in range(64)]
                if len(valid_idxes) > args.num_envs:
                    order = np.argsort(q_value_list)
                    for i, idx in enumerate(order[:64]):
                        current_cdata_copy[i] = STA_YUH.checkpoints[idx][0]
                        vectorized_accumulated_rewards[i] = STA_YUH.checkpoints[idx][3]
                        vectorized_episode_lengths[i] = STA_YUH.checkpoints[idx][4]
                        old_ids[i] = idx
                else:
                    for i, idx in enumerate(valid_idxes):
                        current_cdata_copy[i] = STA_YUH.checkpoints[idx][0]
                        vectorized_accumulated_rewards[i] = STA_YUH.checkpoints[idx][3]
                        vectorized_episode_lengths[i] = STA_YUH.checkpoints[idx][4]
                        old_ids[i] = idx
                
                old_ids = np.array(old_ids)
                env.env.env.env.set_state(current_cdata_copy)
                
                print(f'Training has been augmented with {len(old_ids)} states') # this is not entirely accurate, this prints how many valid states were sampled

        print(f'Time Steps: {total_global_steps}, STA Buffer Size: {len(STA_YUH.checkpoints)}, Best Threshold: {STA_YUH.best_threshold}')
        
        # saving
        torch.save({
            'model': agent.model.state_dict(),
            'optimizer': agent.optimizer.state_dict()
        }, args.save_path + '.pth')
        
    # final save
    torch.save({
            'model': agent.model.state_dict(),
            'optimizer': agent.optimizer.state_dict()
        }, args.save_path + '_final.pth')

    
if __name__ == '__main__':
    args = parse_args()
    main(args)