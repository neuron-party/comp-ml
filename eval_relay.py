import os
import gym
import procgen
import itertools
import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
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
    env.single_action_space = env.action_space.n
    env.single_observation_space = env.observation_space['rgb']
    
    device = torch.device('cuda:' + str(args.device))

    model = ImpalaCNN(env.single_observation_space, env.single_action_space)
    params = {'gamma': 0.999, 'lr': 5e-4, 'epsilon': 0.2, 'vf_clip': 0.2, 'device': device, 'num_steps': 256,
              'num_epochs': 3, 'num_batches': 8, 'num_envs': 128, 'lambd': 0.95, 'c1': 0.5, 'c2': 0.01, 'max_grad_norm': 0.5, 
              'norm_adv': True }
    agent = PPO4(env.single_observation_space.shape, action_space=env.single_action_space, model=model, **params)
    
    agent_pairs = list(itertools.combinations([1, 2, 3, 4], 2))
    
    
    # evaluation loop
    master_failure_list = []
    for idx, agent_pair in enumerate(agent_pairs):
        # we loop twice here since we want both agents to be the main and relay agent
        for i in range(2):
            failures = []
            if i == 0:
                main_agent = torch.load(args.weights_path + str(agent_pair[0]) + '.pth')
                relay_agent = torch.load(args.weights_path + str(agent_pair[1]) + '.pth')
            else:
                main_agent = torch.load(args.weights_path + str(agent_pair[1]) + '.pth')
                relay_agent = torch.load(args.weights_path + str(agent_pair[0]) + '.pth')

            while (len(failures) < 2000):
                t = 0
                state = env.reset()
                agent.model.load_state_dict(main_agent['model'])
                done = verbose = tracked = valid_trajectory = False
                sum_reward = 0

                while not done:
                    action, _, _ = agent.get_action(state)
                    next_state, reward, done, info = env.step(action)
                    state = next_state
                    sum_reward += reward

                    t += 1
                    if t > 400:
                        verbose = valid_trajectory = True
                        agent.model.load_state_dict(relay_agent['model'])
                    if t >= 500 and valid_trajectory and not tracked:
                        failures.append(0)
                        tracked = True

                if t < 500 and valid_trajectory and sum_reward == 0:
                    valid_trajectory = False
                    failures.append(1)
                    
                elif t < 500 and valid_trajectory and sum_reward == 10:
                    failures.append(0)

                if verbose:
                    print(f'Iteration: {idx}, Epochs: {len(failures)}, Time Step: {t}, Outcome: {failures[-1]}, Failure Rate: {sum(failures) / len(failures)}')

            master_failure_list.append(failures) 
            with open(args.save_path + '_checkpoint.pkl', 'wb') as f:
                pickle.dump(master_failure_list, f)
    
    with open(args.save_path + '_final.pkl', 'wb') as f:
        pickle.dump(master_failure_list, f)
        
        
if __name__ == '__main__':
    args = parse_args()
    main(args)