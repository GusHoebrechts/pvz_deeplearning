from collections import deque
import multiprocessing
import os
import select
import sys

import gymnasium as gym
import gym_pvz
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from pvz.display import display_games_grid
from wrapper import PvzMaskWrapper


def _display_training_games(samples):
    games = [sample["frames"] for sample in samples]
    labels = [
        f"Timestep {sample['milestone']:,} | Game {sample['episode']:,}"
        for sample in samples
    ]
    display_games_grid(games, labels=labels)


class RenderTrainingMilestonesCallback(BaseCallback):
    def __init__(self, interval=10_000):
        super().__init__()
        self.interval = interval
        self.next_milestone = 0
        self._pending_milestones = {}
        self._milestone_games = deque(maxlen=4)
        self._render_process = None
        self._render_pending = False

    def _on_step(self):
        self._collect_finished_render()
        for info in self.locals.get("infos", []):
            episode_number = info.get("training_episode_number")
            if episode_number is None:
                continue

            if self.next_milestone == 0:
                self._pending_milestones.setdefault(episode_number, []).append(0)
                self.next_milestone = self.interval
            while self.num_timesteps >= self.next_milestone:
                self._pending_milestones.setdefault(
                    episode_number, []
                ).append(self.next_milestone)
                self.next_milestone += self.interval

            episode_frames = info.get("episode_render_info")
            if episode_frames:
                milestones = self._pending_milestones.pop(episode_number, [])
                for milestone in milestones:
                    self._milestone_games.append(
                        {
                            "milestone": milestone,
                            "episode": episode_number,
                            "frames": episode_frames,
                        }
                    )
                    self._render_pending = True
                    print(
                        f"Captured training game {episode_number} for "
                        f"timestep {milestone:,}."
                    )

        if self._render_pending and self._render_process is None:
            self._start_render()
        return True

    def _start_render(self):
        samples = list(self._milestone_games)
        context = multiprocessing.get_context("spawn")
        self._render_process = context.Process(
            target=_display_training_games,
            args=(samples,),
        )
        self._render_process.start()
        self._render_pending = False

    def _collect_finished_render(self):
        if self._render_process is None or self._render_process.is_alive():
            return

        self._render_process.join()
        exit_code = self._render_process.exitcode
        self._render_process = None
        if exit_code:
            raise RuntimeError(
                f"Training game renderer failed with exit code {exit_code}."
            )

    def _on_training_end(self):
        if self._render_process is not None:
            self._render_process.join()
            self._collect_finished_render()
        if self._render_pending:
            self._start_render()
            self._render_process.join()
            self._collect_finished_render()


class TrainingControlCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.stop_requested = False
        self._terminal_fd = None
        self._terminal_settings = None
        self._interactive = sys.stdin.isatty()
        self._is_windows = os.name == "nt"

    def start(self):
        if not self._interactive:
            print("P/R/Q/Y controls require an interactive terminal.")
            return

        if not self._is_windows:
            import termios
            import tty

            self._terminal_fd = sys.stdin.fileno()
            self._terminal_settings = termios.tcgetattr(self._terminal_fd)
            tty.setcbreak(self._terminal_fd)
            current_settings = termios.tcgetattr(self._terminal_fd)
            current_settings[0] &= ~termios.IXON
            termios.tcsetattr(
                self._terminal_fd, termios.TCSADRAIN, current_settings
            )

        print("Training controls: P = pause, R = resume, Q then Y = stop.")

    def close(self):
        if self._terminal_settings is not None:
            import termios

            termios.tcsetattr(
                self._terminal_fd, termios.TCSADRAIN, self._terminal_settings
            )
            self._terminal_settings = None

    def _read_key(self, wait=False):
        if self._is_windows:
            import msvcrt

            if wait:
                return msvcrt.getwch()
            if msvcrt.kbhit():
                return msvcrt.getwch()
            return None

        timeout = 0.1 if wait else 0
        ready, _, _ = select.select([self._terminal_fd], [], [], timeout)
        if ready:
            return os.read(self._terminal_fd, 1).decode("utf-8", errors="ignore")
        return None

    def _confirm_stop(self):
        print("\nPress Y to confirm stopping, or any other key to cancel.")
        key = self._read_key(wait=True)
        if key is not None and key.lower() == "y":
            self.stop_requested = True
            print("\nTraining stopped. No resumable checkpoint was saved.")
            return True
        print("Stop cancelled.")
        return False

    def _on_step(self):
        if not self._interactive:
            return True

        while True:
            key = self._read_key()
            if key is None:
                return True

            key = key.lower()
            if key == "q" and self._confirm_stop():
                return False
            if key == "p":
                print("\nTraining paused. Press R to resume or Q then Y to stop.")
                while True:
                    key = self._read_key(wait=True)
                    if key is None:
                        continue
                    key = key.lower()
                    if key == "r":
                        print("Training resumed.")
                        break
                    if key == "q" and self._confirm_stop():
                        return False


def main():
    # Directories for checkpoints and logs
    log_dir = "./logs/"
    model_dir = "./saved_models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("Initializing environment...")
    raw_env = gym.make("pvz-env-v2")
    env = PvzMaskWrapper(raw_env, record_episode_frames=True)

    # Evaluation environment (runs in parallel to test greedy performance)
    eval_raw_env = gym.make("pvz-env-v2")
    eval_env = PvzMaskWrapper(eval_raw_env)

    try:
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
        render_callback = RenderTrainingMilestonesCallback(interval=10_000)

        print("Configuring MaskablePPO agent...")
        # Using an MlpPolicy over the flat feature vector
        model = MaskablePPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,        #Number of samples per gradient update. Larger batch sizes can stabilize training but require more memory.
            n_epochs=15,          #how many times to reuse each batch of data for training. More epochs can lead to better performance but also risk overfitting.
            gamma=0.99,           #How much the agent values future rewards compared with immediate rewards. Values closer to 1 make it plan further ahead
            gae_lambda=0.95,      #Controls how PPO estimates future advantage (how good an action was), balancing noisy estimates against bias.
            clip_range=0.2,
            ent_coef=0.05,         # Encourages exploration among valid moves
            verbose=1,
            tensorboard_log=log_dir
        )

        control_callback = TrainingControlCallback()
        try:
            control_callback.start()
            model.learn(
                total_timesteps=1_000_000, #testing with 1 million timesteps for now, can be increased later (+- 30 min run time)
                callback=CallbackList(
                    [eval_callback, render_callback, control_callback]
                ),
                progress_bar=True
            )
        finally:
            control_callback.close()

        if control_callback.stop_requested:
            return

        # Save final model
        final_path = os.path.join(model_dir, "maskable_ppo_pvz_final")
        model.save(final_path)
        print(f"Training complete! Final model saved to: {final_path}")

        # Final policy evaluation
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=10, deterministic=True
        )
        print(f"Mean reward over 10 eval runs: {mean_reward:.2f} +/- {std_reward:.2f}")
    finally:
        env.close()
        eval_env.close()

if __name__ == "__main__":
    main()