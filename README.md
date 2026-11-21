# Plants vs. Zombies RL Project

This project trains Reinforcement Learning agents to play a Plants vs. Zombies clone using modern frameworks (Gymnasium + Stable-Baselines3).

## Current Workspace
* **`pvz_core/`**: The core game engine and configurations (graphics, balancing, entity behavior, FPS).
* **`gym-pvz/`**: The environment wrapper that makes the game engine compatible with the modern `gymnasium` RL API.
* **`train.py`**: Our new baseline training script utilizing `MaskablePPO` to handle invalid action masking (e.g., preventing planting on occupied tiles or without enough sun).
* **`train_a2c.py`**: An A2C baseline with action masking, using the same environment, observation, reward, and timestep budget.

## Setup Instructions

1. **Clone the repository:**
   git clone <your-repo-link>
   cd <your-repo-folder>
   
2. **Install the game engine and environment:**
 Install both local libraries in editable mode so any changes to the game engine take effect immediately:

pip install -e ./pvz_core
pip install -e ./gym-pvz

3. **Install the RL toolchain:**

pip install gymnasium stable-baselines3 sb3-contrib wandb tensorboard tqdm rich

`pygame-ce` (which provides the `pygame` import) is installed as a dependency
of `pvz_core` and is used by `watch.py` to replay the recorded game frames.

4. **Test the environment:**

python test.py

## Training

Start training with:

python train.py

The agent observation includes each plant's remaining cooldown as a continuous
fraction (1 = full cooldown, 0 = ready), in addition to the existing
ready/affordable indicator. It also exposes each potato mine's arming progress
per tile (0 = unarmed, 1 = ready to explode), and the zombie count on each tile
alongside the total zombie HP there. These additions change the observation
shape, so previously trained models must be retrained.

Configured zombie scores are awarded gradually in proportion to actual HP
damage, rather than as an additional bonus on death.

During training, press `P` to pause, `R` to resume, or press `Q` followed by
`Y` to stop without saving a resumable checkpoint. These controls require an
interactive terminal.

Run `python watch.py` to view the best evaluation checkpoint. Use
`python watch.py --model final` to view the final training checkpoint instead.

## A2C comparison

Run `python train_a2c.py` to train the masked A2C baseline. Its checkpoints
and TensorBoard logs are kept separate from PPO under `saved_models/a2c/` and
`logs/a2c/`. Run `python watch_a2c.py` to replay the best evaluation
checkpoint, or `python watch_a2c.py --model final` to replay the final model.

A2C uses its own masking-aware policy, rollout buffer, and environment wrapper;
it does not change the PPO implementation or shared environment code. Both
agents therefore receive the same legal-action mask during training and
evaluation.

## Comparing trained models

After training both agents, run `python compare_models.py` to evaluate their
best checkpoints on 100 identical environment seeds. Select checkpoints and
evaluation size with `--ppo-checkpoint`, `--a2c-checkpoint`, `--episodes`, and
`--seed-start`. The report includes mean/median reward, survival rate,
bootstrap confidence intervals, paired reward and survival differences, and
invalid-action counts. Per-episode results, summary CSVs, metadata, and a
Markdown report are written to a timestamped directory under
`comparison_results/`. Checkpoints must match the current environment's
observation and action spaces; the script stops with a clear error if a
checkpoint needs to be retrained.

The highest-reward completed training game in each 10,000-timestep window is
replayed in a rolling four-game Pygame grid. Each tile shows the absolute
checkpoint timestep, the window it covers, the actual training game number,
and episode reward. Recordings are displayed in a separate process while
training continues. The entropy coefficient decays linearly from 0.05 to 0.01
over the 1,000,000 training timesteps. Close the window or press `Esc` to
dismiss it.

Credits
The core game engine and the initial environment framework were originally built by inf581-pvz-anonymous. We have heavily refactored the environment to support the modern gymnasium API and are implementing entirely new agents using sb3-contrib and supercomputer batch processing.
