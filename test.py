import gymnasium as gym
import gym_pvz

# Initialize the environment
env = gym.make("pvz-env-v2")

# Test the reset
obs, info = env.reset()
print(f"Environment successfully loaded! Observation shape: {obs.shape}")

env.close()