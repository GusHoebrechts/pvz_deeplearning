import gymnasium as gym
import gym_pvz

# Initialize the environment
env = gym.make('gym_pvz:pvz-env-v2')

# Test the reset
obs, info = env.reset()
print(f"Environment successfully loaded! Observation shape: {obs.shape}")

env.close()