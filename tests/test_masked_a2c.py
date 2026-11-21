import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

from a2c_masking import A2CMaskWrapper, MaskedA2C


class MaskedToyEnv(gym.Env):
    observation_space = gym.spaces.Box(
        low=0, high=1, shape=(1,), dtype=np.float32
    )
    action_space = gym.spaces.Discrete(3)

    def __init__(self):
        self.state = 0
        self.steps = 0
        self.invalid_action_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        self.steps = 0
        return np.array([self.state], dtype=np.float32), {}

    def action_masks(self):
        return np.array([True, self.state == 0, self.state == 1])

    def mask_available_actions(self):
        return self.action_masks()

    def step(self, action):
        mask = self.action_masks()
        if not mask[int(action)]:
            self.invalid_action_count += 1

        reward = float(action == (1 if self.state == 0 else 2))
        self.state = 1 - self.state
        self.steps += 1
        terminated = self.steps >= 4
        return (
            np.array([self.state], dtype=np.float32),
            reward,
            terminated,
            False,
            {},
        )


class MaskedA2CTests(unittest.TestCase):
    def test_training_prediction_and_evaluation_respect_action_masks(self):
        training_base_env = A2CMaskWrapper(MaskedToyEnv())
        training_env = Monitor(training_base_env)
        evaluation_base_env = MaskedToyEnv()
        evaluation_env = Monitor(A2CMaskWrapper(evaluation_base_env))
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                model = MaskedA2C(
                    "MlpPolicy",
                    training_env,
                    n_steps=4,
                    seed=0,
                    verbose=0,
                )
                observation, _ = training_env.reset(seed=0)
                action_mask = training_base_env.action_masks()
                action, _ = model.predict(
                    observation,
                    action_masks=action_mask,
                    deterministic=True,
                )
                self.assertTrue(action_mask[int(action)])
                with self.assertRaises(ValueError):
                    model.predict(observation, deterministic=True)

                eval_callback = MaskableEvalCallback(
                    evaluation_env,
                    best_model_save_path=temp_dir,
                    eval_freq=4,
                    n_eval_episodes=2,
                    deterministic=True,
                    verbose=0,
                )
                model.learn(total_timesteps=8, callback=eval_callback)

                self.assertEqual(
                    model.get_env().get_attr("invalid_action_count")[0], 0
                )
                self.assertEqual(evaluation_base_env.invalid_action_count, 0)
                self.assertTrue((Path(temp_dir) / "best_model.zip").is_file())

                model_path = Path(temp_dir) / "masked_a2c"
                model.save(model_path)
                loaded_model = MaskedA2C.load(model_path)
                loaded_action, _ = loaded_model.predict(
                    observation,
                    action_masks=action_mask,
                    deterministic=True,
                )
                self.assertTrue(action_mask[int(loaded_action)])
        finally:
            training_env.close()
            evaluation_env.close()


if __name__ == "__main__":
    unittest.main()
