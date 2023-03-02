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

from models.ppg import *
from models.ppg_policy import *


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
    parser.add_argument('--batch-norm-adv', type=bool, default=True)
    parser.add_argument('--n-pi', type=int, default=32)
    parser.add_argument('--e-aux', type=int, default=6)
    parser.add_argument('--num-aux-rollouts', type=int, default=8)
    parser.add_argument('--beta-clone', type=float, default=1.0)
    
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
    
    args = parser.parse_args()
    return args

def main(args):
    if args.log:
        wandb.init(
            project=args.project_name,
            name=args.instance_name,
            sync_tensorboard=True,
            monitor_gym=True
        )
        
    writer = SummaryWriter(f'runs/{args.instance_name}')
    
    envs = procgen.ProcgenEnv(
        num_envs=args.num_envs, 
        env_name=args.env_name,
        num_levels=args.num_levels, 
        start_level=args.start_level, 
        distribution_mode=args.distribution_mode)
    
    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space["rgb"]
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.NormalizeReward(envs, gamma=args.gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    
    model = PPG_Impala(envs)
    
    device = torch.device('cuda:' + str(args.device))
    
    params = {
        'gamma': args.gamma, 'lr': args.lr, 'epsilon': args.epsilon, 'vf_clip': args.vf_clip, 'device': device, 'num_steps': args.num_steps,
        'num_epochs': args.num_epochs, 'num_batches': args.num_batches, 'num_envs': args.num_envs, 'lambd': args.lambd, 'n_pi': args.n_pi, 'e_aux': args.e_aux, 
        'c1': args.c1, 'c2': args.c2, 'max_grad_norm': args.max_grad_norm, 'batch_norm_adv': args.batch_norm_adv, 'beta_clone': args.beta_clone, 'num_aux_rollouts': args.num_aux_rollouts
    }
    
    agent = PPG(envs.single_observation_space.shape, envs.single_action_space.n, model, **params)
    
    # training loop
    max_global_steps = args.max_global_steps
    total_global_steps = 0
    state = envs.reset()
    
    while total_global_steps < max_global_steps:
        for i in range(args.n_pi):
            for i in range(args.num_steps):
                action, log_probs, value_estimate, _ = agent.get_action(state)
                next_state, reward, done, info = envs.step(action)
                agent.remember(state, action, reward, done, log_probs, value_estimate, i)
                
                state = next_state
                total_global_steps += args.num_envs
                
                for item in info:
                    if 'episode' in item.keys():
                        writer.add_scalar('charts/episodic_return', item['episode']['r'], total_global_steps)
                        
            agent.policy_phase(state)
        agent.auxiliary_phase()
        
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