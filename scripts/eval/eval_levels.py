import numpy as np
import torch
import torch.nn as nn
import gym
import procgen
import pickle
import argparse

from models.policies import *
from models.ppo import *
from models.ppg import *
from models.ppg_policy import *
from utils.env_utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent-type', type=str, default='ppo')
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--num-envs', type=int, default=100)
    parser.add_argument('--env-name', type=str, default='jumper')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num-levels', type=int, default=500)
    parser.add_argument('--weights-path', type=str, default='weights/ppo_jumper_1')
    parser.add_argument('--save-path', type=str, default='weights/metrics/...')
    
    args = parser.parse_args()
    return args

    
def main(args):
    device = torch.device('cuda:' + str(args.device))
    
    env = procgen.ProcgenEnv(
        num_envs=args.num_envs,
        env_name=args.env_name,
        num_levels=500,
        start_level=0,
        distribution_mode='hard'
    )
    env = gym.wrappers.TransformObservation(env, lambda obs: obs['rgb'])
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.is_vector_env = True
    env.single_action_space = env.action_space
    env.single_observation_space = env.observation_space['rgb']
    
    if args.agent_type == 'ppo':
        model = ImpalaCNN(env.single_observation_space, env.single_action_space.n)
        params = {'gamma': 0.999, 'lr': 5e-4, 'epsilon': 0.2, 'vf_clip': 0.2, 'device': device, 'num_steps': 256,
                  'num_epochs': 3, 'num_batches': 8, 'num_envs': args.num_envs, 'lambd': 0.95, 'c1': 0.5, 'c2': 0.01, 'max_grad_norm': 0.5, 
                  'norm_adv': True }
        agent = PPO4(env.single_observation_space.shape, action_space=env.single_action_space.n, model=model, **params)
        
    elif args.agent_type == 'ppg':
        model = PPG_Impala(env)
        params = {
            'gamma': 0.999, 'lr': 5e-4, 'epsilon': 0.2, 'vf_clip': 0.2, 'device': device, 'num_steps': 256,
            'num_epochs': 3, 'num_batches': 8, 'num_envs': args.num_envs, 'lambd': 0.95, 'n_pi': 32, 'e_aux': 6, 
            'c1': 0.5, 'c2': 0.01, 'max_grad_norm': 0.5, 'batch_norm_adv': True, 'beta_clone': 1.0, 'num_aux_rollouts': 8
        }
        agent = PPG(env.single_observation_space.shape, env.single_action_space.n, model, **params)
        
    else:
        raise ValueError('other agents not implemented yet')
    
    # load model
    weights = torch.load(args.weights_path + '.pth')
    agent.model.load_state_dict(weights['model'])
    
    # eval
    all_level_returns = []
    for i in range(args.num_levels): # evaluate performance on all levels it was trained on
        env = initialize_env(
            num_envs=args.num_envs,
            env_name=args.env_name,
            num_levels=1,
            start_level=i,
            distribution_mode='hard'
        )
        
        returns = rollout_trajectories(agent, env, n=args.n, agent_type=args.agent_type)
        print(f'Level: {i}, Average Return: {np.mean(returns)}')
        all_level_returns.append(returns)
        
    with open(args.save_path + '.pkl', 'wb') as f:
        pickle.dump(all_level_returns, f)
        

if __name__ == '__main__':
    args = parse_args()
    main(args)