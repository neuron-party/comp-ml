import os
import gym
import wandb
import metadrive
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

from models.ppo5 import *
from models.policies import *


def parse_args():
    parser = argparse.ArgumentParser()
    
    # agent args
    # default parameters for PPO can be found on page 9 of https://arxiv.org/pdf/2109.12674.pdf
    # tweaked some of the default params since it wasnt performing well
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--vf-clip', type=float, default=0.2)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num-steps', type=int, default=256)
    # parser.add_argument('--num-epochs', type=int, default=20) # why so many optimization epochs? procgen uses 3, also too many epochs overfits to the most recent sample?
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--num-batches', type=int, default=64)
    parser.add_argument('--lambd', type=float, default=0.95)
    parser.add_argument('--c1', type=float, default=0.5)
    parser.add_argument('--c2', type=float, default=0.01)
    parser.add_argument('--max-grad-norm', type=float, default=False)
    parser.add_argument('--norm-adv', type=bool, default=True)
    
    # env args
    parser.add_argument('--env-name', type=str, default='MetaDrive-100envs-v0')
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--max-global-steps', type=int, default=100000000)
    
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
    
    env = gym.vector.make(args.env_name, num_envs=args.num_envs)
    
    model = ContinuousActionMLP(observation_space=259, action_space=2)
    
    device = torch.device('cuda:' + str(args.device))
    params = {
        'gamma': args.gamma, 'lr': args.lr, 'epsilon': args.epsilon, 'vf_clip': args.vf_clip, 'device': device, 'num_steps': args.num_steps,
        'num_epochs': args.num_epochs, 'num_batches': args.num_batches, 'num_envs': args.num_envs, 'lambd': args.lambd, 
        'c1': args.c1, 'c2': args.c2, 'max_grad_norm': args.max_grad_norm, 'norm_adv': args.norm_adv
    }
    
    agent = PPO5(observation_space=(259, ), action_space=2, model=model, **params)
    
    # training loop
    max_global_steps = args.max_global_steps
    total_global_steps = 0
    state = env.reset()
    
    while total_global_steps < max_global_steps:
        for i in range(args.num_steps):
            action, log_probs, value_estimate = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, done, log_probs, value_estimate, i)
            
            state = next_state
            total_global_steps += args.num_envs
            
            for single_done, single_info in zip(done, info):
                if single_done:
                    # episode return
                    writer.add_scalar('charts/episodic_return', single_info['episode_reward'], total_global_steps)
                    # success/failure
                    if single_info['arrive_dest']:
                        writer.add_scalar('charts/success_rate', 1, total_global_steps)
                    else:
                        writer.add_scalar('charts/success_rate', 0, total_global_steps)
        
        agent.learn(state)
        
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