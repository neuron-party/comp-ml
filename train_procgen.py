import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gym
import procgen
from torch.utils.tensorboard import SummaryWriter
import wandb
import argparse

from models.ppo import *
from models.policies import *


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
        'c1': args.c1, 'c2': args.c2, 'max_grad_norm': args.max_grad_norm
    }
    
    agent = PPO4(env.single_observation_space.shape, env.single_action_space, model, **params)
    
    # training loop
    max_global_steps = args.max_global_steps
    total_global_steps = 0
    state = env.reset()
    
    while total_global_steps < max_global_steps:
        for i in range(args.num_steps):
            action, log_probs, value_estimate = agent.get_action(state)
            next_state, reward, done, info = envs.step(action)
            agent.remember(state, action, reward, done, log_probs, value_estimate, i)
            state = next_state
            total_global_steps += args.num_envs
            for item in info:
                if 'episode' in item.keys():
                    episodic_returns.append(item['episode']['r'])
                    writer.add_scalar('charts/episodic_return', item['episode']['r'], total_steps)
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