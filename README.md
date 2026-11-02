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

pip install gymnasium stable-baselines3 sb3-contrib wandb

`pygame-ce` (which provides the `pygame` import) is installed as a dependency
of `pvz_core` and is used by `watch.py` to replay the recorded game frames.

4. **Test the environment:**

python test.py

Credits
The core game engine and the initial environment framework were originally built by inf581-pvz-anonymous. We have heavily refactored the environment to support the modern gymnasium API and are implementing entirely new agents using sb3-contrib and supercomputer batch processing.
