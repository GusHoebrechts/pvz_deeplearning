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
from stable_baselines3.common.monitor import Monitor
from pvz.display import display_games_grid
from wrapper import PvzMaskWrapper

TOTAL_TIMESTEPS = 1_000_000
ENTROPY_COEF_START = 0.05
ENTROPY_COEF_END = 0.01


def entropy_coefficient(progress_remaining):
    progress_remaining = min(max(progress_remaining, 0.0), 1.0)
    return ENTROPY_COEF_END + (
        ENTROPY_COEF_START - ENTROPY_COEF_END
    ) * progress_remaining


def _training_game_label(sample):
    if "window_start" in sample:
        return (
            f"Checkpoint {sample['window_end']:,} "
            f"| Window {sample['window_start']:,}-{sample['window_end']:,} "
            f"| Training game #{sample['episode']:,} "
            f"| Reward {sample['reward']:.0f}"
        )
    return (
        f"Checkpoint {sample['milestone']:,} "
        f"| Training game #{sample['episode']:,}"
    )


def _display_training_games(samples):
    games = [sample["frames"] for sample in samples]
    labels = [_training_game_label(sample) for sample in samples]
    display_games_grid(games, labels=labels)


class RenderTrainingMilestonesCallback(BaseCallback):
    def __init__(self, interval=10_000):
        super().__init__()
        self.interval = interval
        self.window_start = 0
        self.next_milestone = interval
        self._best_run = None
        self._milestone_games = deque(maxlen=4)
        self._render_process = None
        self._render_pending = False

    def _record_episode(self, episode_number, reward, episode_frames):
        if self._best_run is None or reward > self._best_run["reward"]:
            self._best_run = {
                "episode": episode_number,
                "reward": reward,
                "frames": episode_frames,
                "window_start": self.window_start,
            }

    def _finish_window(self, window_end):
        if self._best_run is not None:
            sample = dict(self._best_run)
            sample["window_end"] = window_end
            self._milestone_games.append(sample)
            self._render_pending = True
            print(
                f"Best training game #{sample['episode']:,} for checkpoint "
                f"{window_end:,} (window {self.window_start:,}-{window_end:,}), "
                f"reward {sample['reward']:.0f}."
            )

        self._best_run = None
        self.window_start = window_end

    def _on_step(self):
        self._collect_finished_render()
        for info in self.locals.get("infos", []):
            episode_number = info.get("training_episode_number")
            episode_frames = info.get("episode_render_info")
            if episode_frames and episode_number is not None:
                episode_stats = info.get("episode")
                if episode_stats is None or "r" not in episode_stats:
                    raise RuntimeError(
                        "Training episodes must be monitored to select the "
                        "best run by reward."
                    )
                self._record_episode(
                    episode_number,
                    episode_stats["r"],
                    episode_frames,
                )

        while self.num_timesteps >= self.next_milestone:
            self._finish_window(self.next_milestone)
            self.next_milestone += self.interval

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
            print(
                "Warning: training game renderer failed with "
                f"exit code {exit_code}; training will continue."
            )

    def _on_training_end(self):
        if self._best_run is not None:
            self._finish_window(self.num_timesteps)
        if self._render_process is not None:
            self._render_process.join()
            self._collect_finished_render()
        if self._render_pending:
            self._start_render()
            self._render_process.join()
            self._collect_finished_render()


class EntropyCoefficientScheduleCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps

    def _on_step(self):
        progress = min(self.num_timesteps / self.total_timesteps, 1.0)
        self.model.ent_coef = entropy_coefficient(1.0 - progress)
        return True

    def _on_rollout_end(self):
        self.logger.record("train/ent_coef", self.model.ent_coef)


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
            print("\nStop confirmed; saving the current model.")
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
    env = PvzMaskWrapper(
        Monitor(raw_env),
        record_episode_frames=True,
    )

    # Evaluation environment (runs in parallel to test greedy performance)
    eval_raw_env = gym.make("pvz-env-v2")
    eval_env = PvzMaskWrapper(Monitor(eval_raw_env))

    try:
        # Automatically save the best model during training
        eval_callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            log_path=log_dir,
            eval_freq=5000,
            n_eval_episodes=40,
            deterministic=True,
            render=False
        )
        render_callback = RenderTrainingMilestonesCallback(interval=10_000)
        entropy_callback = EntropyCoefficientScheduleCallback(TOTAL_TIMESTEPS)

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
                total_timesteps=TOTAL_TIMESTEPS, #testing with 1 million timesteps for now, can be increased later (+- 30 min run time)
                callback=CallbackList(
                    [
                        eval_callback,
                        render_callback,
                        entropy_callback,
                        control_callback,
                    ]
                ),
                progress_bar=True
            )
        finally:
            control_callback.close()

        final_path = os.path.join(model_dir, "maskable_ppo_pvz_final")
        model.save(final_path)
        print(f"Final model saved to: {final_path}")

        if control_callback.stop_requested:
            print("Training stopped. The best evaluation model is preserved.")
            return

        print("Training complete!")

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