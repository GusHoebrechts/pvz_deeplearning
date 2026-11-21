# Plants vs. Zombies RL Project

This project trains Reinforcement Learning agents to play a Plants vs. Zombies clone using modern frameworks (Gymnasium + Stable-Baselines3).

## Current Workspace
* **`pvz_core/`**: The core game engine and configurations (graphics, balancing, entity behavior, FPS).
* **`gym-pvz/`**: The environment wrapper that makes the game engine compatible with the modern `gymnasium` RL API.
* **`train.py`**: Our new baseline training script utilizing `MaskablePPO` to handle invalid action masking (e.g., preventing planting on occupied tiles or without enough sun).

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
ready/affordable indicator. This changes the observation shape, so previously
trained models must be retrained.

During training, press `P` to pause, `R` to resume, or press `Q` followed by
`Y` to stop without saving a resumable checkpoint. These controls require an
interactive terminal.

Training games selected at timesteps 0, 10,000, 20,000, 30,000, and every
10,000 timesteps afterward are replayed together in a rolling four-game Pygame
grid. Each tile shows its checkpoint timestep and actual training game number.
The recordings are displayed in a separate process while training continues;
games appear after their episode completes. Close the window or press `Esc` to
dismiss it.

Credits
The core game engine and the initial environment framework were originally built by inf581-pvz-anonymous. We have heavily refactored the environment to support the modern gymnasium API and are implementing entirely new agents using sb3-contrib and supercomputer batch processing.
