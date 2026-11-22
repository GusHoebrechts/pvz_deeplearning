import os

import gymnasium as gym
import gym_pvz
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from stable_baselines3.common.monitor import Monitor

from a2c_masking import A2CMaskWrapper, MaskedA2C


TOTAL_TIMESTEPS = 1_000_000
EVAL_FREQUENCY = 5_000
EVAL_EPISODES = 40


def main():
    log_dir = "./logs/a2c/"
    model_dir = "./saved_models/a2c/"
    eval_log_dir = os.path.join(log_dir, "evaluations")
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    env = Monitor(A2CMaskWrapper(gym.make("pvz-env-v2")))
    eval_env = Monitor(A2CMaskWrapper(gym.make("pvz-env-v2")))

    try:
        eval_callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            log_path=eval_log_dir,
            eval_freq=EVAL_FREQUENCY,
            n_eval_episodes=EVAL_EPISODES,
            deterministic=True,
        )

        model = MaskedA2C(
            policy="MlpPolicy",
            env=env,
            rollout_buffer_class=MaskableRolloutBuffer,
            learning_rate=7e-4,
            n_steps=128,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_rms_prop=True,
            verbose=1,
            tensorboard_log=log_dir,
        )

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=eval_callback,
            progress_bar=True,
        )

        final_path = os.path.join(model_dir, "a2c_pvz_final")
        model.save(final_path)
        print(f"Final model saved to: {final_path}")

        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=10,
            deterministic=True,
            use_masking=True,
        )
        print(
            f"Mean reward over 10 eval runs: "
            f"{mean_reward:.2f} +/- {std_reward:.2f}"
        )
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
