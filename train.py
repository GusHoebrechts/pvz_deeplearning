import os
import gymnasium as gym
import gym_pvz
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from wrapper import PvzMaskWrapper

def main():
    # Directories for checkpoints and logs
    log_dir = "./logs/"
    model_dir = "./saved_models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("Initializing environment...")
    raw_env = gym.make('gym_pvz:pvz-env-v2')
    env = PvzMaskWrapper(raw_env)

    # Evaluation environment (runs in parallel to test greedy performance)
    eval_raw_env = gym.make('gym_pvz:pvz-env-v2')
    eval_env = PvzMaskWrapper(eval_raw_env)

    # Automatically save the best model during training
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    print("Configuring MaskablePPO agent...")
    # Using an MlpPolicy over the flat feature vector
    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # Encourages exploration among valid moves
        verbose=1,
        tensorboard_log=log_dir
    )

    print("Starting training run...")
    total_timesteps = 100_000   # Good smoke test on CPU; scale to 1M+ on the supercomputer
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )

    # Save final model
    final_path = os.path.join(model_dir, "maskable_ppo_pvz_final")
    model.save(final_path)
    print(f"Training complete! Final model saved to: {final_path}")

    # Final policy evaluation
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"Mean reward over 10 eval runs: {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()