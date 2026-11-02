from pathlib import Path

import gymnasium as gym
import gym_pvz
from sb3_contrib import MaskablePPO
from wrapper import PvzMaskWrapper
from pvz.display import display_game


ROOT_DIR = Path(__file__).resolve().parent


def main():
    model_path = ROOT_DIR / "saved_models" / "best_model.zip"
    if not model_path.exists():
        model_path = ROOT_DIR / "saved_models" / "maskable_ppo_pvz_final.zip"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found in {ROOT_DIR / 'saved_models'}. "
            "Train the model first using train.py."
        )

    print(f"Loading policy weights from: {model_path}")
    model = MaskablePPO.load(str(model_path))

    raw_env = gym.make("pvz-env-v2")
    env = PvzMaskWrapper(raw_env)

    num_episodes = 3
    try:
        for ep in range(1, num_episodes + 1):
            obs, info = env.reset()
            done = False
            total_reward = 0
            step_count = 0

            print(f"\n--- Simulating Episode {ep} ---")

            while not done:
                action_masks = env.action_masks()
                action, _ = model.predict(
                    obs,
                    action_masks=action_masks,
                    deterministic=True
                )

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                total_reward += reward
                step_count += 1

            print(
                f"Episode {ep} finished: Decision steps = {step_count}, "
                f"Total reward = {total_reward:.2f}"
            )

            render_info = env.unwrapped.get_render_info()

            if render_info:
                print(
                    f"Opening Pygame window: replaying {len(render_info)} "
                    "animation frames..."
                )
                if not display_game(render_info):
                    break
            else:
                print("Warning: render_info was empty.")
    finally:
        env.close()


if __name__ == "__main__":
    main()