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
Install the game engine and environment:
Install both local libraries in editable mode so any changes to the game engine take effect immediately:

pip install -e ./pvz_core
pip install -e ./gym-pvz
Install the RL toolchain:

pip install stable-baselines3 sb3-contrib wandb
Test the environment:

python train.py
