import gymnasium as gym
import numpy as np

class PvzMaskWrapper(gym.Wrapper):
    """
    Wraps the custom PvZ environment to expose the action_masks()
    method required by sb3-contrib's MaskablePPO.
    """
    def __init__(self, env, record_episode_frames=False):
        super().__init__(env)
        self.record_episode_frames = record_episode_frames
        self.episode_number = -1

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.episode_number += 1
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.record_episode_frames:
            info = dict(info)
            info["training_episode_number"] = self.episode_number
            if terminated or truncated:
                info["episode_render_info"] = self.env.unwrapped.get_render_info()
        return observation, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        # Access mask_available_actions from the unwrapped core game
        raw_mask = self.env.unwrapped.mask_available_actions()
        return np.array(raw_mask, dtype=bool)