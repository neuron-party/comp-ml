import gym
import procgen

def initialize_env(num_envs, env_name, num_levels=500, start_level=0, distribution_mode='hard'):
    env = procgen.ProcgenEnv(
        num_envs=num_envs,
        env_name=env_name,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode
    )
    env = gym.wrappers.TransformObservation(env, lambda obs: obs['rgb'])
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.is_vector_env = True
    env.single_action_space = env.action_space
    env.single_observation_space = env.observation_space['rgb']
    
    return env


def rollout_trajectories(agent, env, n=500, agent_type='ppo'):
    returns = []
    
    state = env.reset()
    
    while len(returns) < n:
        if agent_type == 'ppo':
            action, _, _ = agent.get_action(state)
        elif agent_type == 'ppg':
            action, _, _, _ = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        
        for item in info:
            if 'episode' in item.keys():
                returns.append(item['episode']['r'])
    
    return returns