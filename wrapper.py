import gymnasium as gym
import numpy as np

class PvzMaskWrapper(gym.Wrapper):
    """
    Wraps the custom PvZ environment to expose the action_masks()
    method required by sb3-contrib's MaskablePPO.
    """
    def __init__(self, env):
        super().__init__(env)

    def action_masks(self) -> np.ndarray:
        # Access mask_available_actions from the unwrapped core game
        raw_mask = self.env.unwrapped.mask_available_actions()
        return np.array(raw_mask, dtype=bool)