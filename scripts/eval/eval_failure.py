# more generalized script for same agent failures
import os
import gym
import procgen
import itertools
import argparse
import pickle
import torch
import torch.nn as nn
import numpy as np

from models.ppg_policy import *
from models.ppg import *
from models.ppo import *
from models.policies import *


def parse_args():
    parser = argparse.ArgumentParser()
    
    # no need to add argument parsers for agent parameters
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--env-name', type=str, default='jumper')
    parser.add_argument('--num-levels', type=int, default=500)
    parser.add_argument('--start-level', type=int, default=0)
    parser.add_argument('--distribution-mode', type=str, default='hard')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--save-path', type=str, default='list')
    parser.add_argument('--weights-path', type=str, default='weights/ppo_jumper_')
    
    parser.add_argument('--environment-type', type=str, default='completion') # completion or continuous (i.e jumper vs starpilot) type of environments
    parser.add_argument('--agent-type', type=str, default='ppo') # ppo or ppg
    parser.add_argument('--num-agents', type=int, default=3) # number of separately trained agents
    
    args = parser.parse_args()
    return args


def main(args):
    env = procgen.ProcgenEnv(
        num_envs=args.num_envs,
        env_name=args.env_name,
        num_levels=args.num_levels,
        start_level=args.start_level,
        distribution_mode=args.distribution_mode
    )
    env = gym.wrappers.TransformObservation(env, lambda obs: obs['rgb'])
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.is_vector_env = True
    env.single_action_space = env.action_space
    env.single_observation_space = env.observation_space['rgb']
    
    device = torch.device('cuda:' + str(args.device))
    
    if args.agent_type == 'ppo':
        model = ImpalaCNN(env.single_observation_space, env.single_action_space.n)
        # default parameters
        params = {'gamma': 0.999, 'lr': 5e-4, 'epsilon': 0.2, 'vf_clip': 0.2, 'device': device, 'num_steps': 256,
                  'num_epochs': 3, 'num_batches': 8, 'num_envs': 128, 'lambd': 0.95, 'c1': 0.5, 'c2': 0.01, 'max_grad_norm': 0.5, 
                  'norm_adv': True }
        agent = PPO4(env.single_observation_space.shape, action_space=env.single_action_space.n, model=model, **params)    
        
    elif args.agent_type == 'ppg':
        model = PPG_Impala(env)
        # default parameters
        params = {
            'gamma': 0.999, 'lr': 5e-4, 'epsilon': 0.2, 'vf_clip': 0.2, 'device': device, 'num_steps': 256,
            'num_epochs': 3, 'num_batches': 8, 'num_envs': args.num_envs, 'lambd': 0.95, 'n_pi': 32, 'e_aux': 6, 
            'c1': 0.5, 'c2': 0.01, 'max_grad_norm': 0.5, 'batch_norm_adv': True, 'beta_clone': 1.0, 'num_aux_rollouts': 8
        }
        agent = PPG(env.single_observation_space.shape, env.single_action_space.n, model, **params)
        
    else:
        raise ValueError('other agent algorithms not implemented yet')
    
    # evaluation loop
    master_failure_list = []
    for i in range(1, args.num_agents + 1):
        agent_weights = torch.load(args.weights_path + str(i) + '.pth')
        agent.model.load_state_dict(agent_weights['model'])
        
        failures = []
        while (len(failures) < 2000):
            t = 0
            state = env.reset()
            done = verbose = tracked = valid_trajectory = False
            sum_reward = 0
            
            while not done:
                if args.agent_type == 'ppo':
                    action, _, _ = agent.get_action(state)
                elif args.agent_type == 'ppg':
                    action, _, _, _ = agent.get_action(state)
                else:
                    raise ValueError('other agent algorithms not implemented yet')
                    
                next_state, reward, done, info = env.step(action)
                state = next_state
                sum_reward += reward

                t += 1
                if t > 400:
                    verbose = valid_trajectory = True
                if t >= 500 and valid_trajectory and not tracked:
                    failures.append(0)
                    tracked = True
            
            # completion type of environments; if the agent completes the level between [400, 500], we still count it as a success, otherwise failure
            if args.environment_type == 'completion':
                if t < 500 and valid_trajectory and sum_reward == 0:
                    valid_trajectory = False
                    failures.append(1)

                elif t < 500 and valid_trajectory and sum_reward == 10:
                    failures.append(0)
                
            # continuous type of environments that can be played forever; if the agent dies within [400, 500], then failure
            elif args.environment_type == 'continuous':
                if t < 500 and valid_trajectory:
                    valid_trajectory = False
                    failures.append(1)

            if verbose:
                print(f'Iteration: {i - 1}, Epochs: {len(failures)}, Time Step: {t}, Outcome: {failures[-1]}, Failure Rate: {sum(failures) / len(failures)}')

        master_failure_list.append(failures) 
        with open(args.save_path + '_checkpoint.pkl', 'wb') as f:
            pickle.dump(master_failure_list, f)
    
    with open(args.save_path + '_final.pkl', 'wb') as f:
        pickle.dump(master_failure_list, f)


        
if __name__ == '__main__':
    args = parse_args()
    main(args)