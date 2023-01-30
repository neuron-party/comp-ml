import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import gymnasium as gym
import procgen
import random
import copy
from collections import deque

# need to fix these imports
from procgen import ProcgenEnv
from vec_env import VecExtractDictObs
from vec_env import VecMonitor
from vec_env import VecNormalize
from util import logger

from ppo import *
from policies import *


# hyperparameters (move to cfg file)
lr = 5e-4
c1 = 0.5
c2 = 0.01
gamma = 0.999
nsteps = 256
num_batches = 8
epochs = 3
epsilon = 0.2
batch_size = 64


def main(): # parse a config file for parameters
    save_path = '../weights/ppo_starpilot_impalacnn.pth'
    params = {
        'lr': lr, 'c1': c1, 'c2': c2, 'gamma': gamma, 'num_batches': 8, 'epochs': 3, 'device': torch.device('cuda:0')
    }
    
    num_envs = 16
    env_name = 'starpilot'
    num_levels = 0
    start_level = 0
    distribution_mode = 'hard'
    num_threads = 1
    is_valid = False
    
    venv = ProcgenEnv(
            num_envs=num_envs,
            env_name=env_name,
            num_levels=0 if is_valid else num_levels,
            start_level=0 if is_valid else start_level,
            distribution_mode=distribution_mode,
            num_threads=num_threads,
    )
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    
    observation_space = venv.observation_space
    action_space = venv.action_space.n
    
    agent = PPO(observation_space, action_space, **params)
    
    max_steps = 200000000 # 200 million steps
    total_steps = 0
    eps = 0
    avg_rewards = [] # i should log these in a csv
    
    while total_steps < max_steps:
        done = False
        state = venv.reset()
        sum_reward = 0
        for i in range(num_steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = venv.step(action)
            agent.remember(reward, done)
            
            state = next_state
            sum_reward += reward
            total_steps += 1
            
            if done:
                break
        
        eps += 1
        avg_rewards.append(np.mean(sum_reward))
        
        if eps % 50 == 0: 
            print(f'Steps Taken: {total_steps}, Episodes: {eps}, Average Reward: {np.mean(avg_rewards)}, Last Episode Reward: {sum_reward}')
            torch.save({
                'model': agent.policy.state_dict(),
                'optimizer': agent.optimizer.state_dict()
            }, save_path)