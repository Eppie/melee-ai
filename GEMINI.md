# Nano-Melee

## Project Overview

This project, "Nano-Melee," is designed to train a machine learning model to play the video game Super Smash Bros. Melee. It utilizes a GPT-style transformer model to learn and execute complex controller inputs.

**Key Technologies:**
- **Programming Language:** Python 3.12+
- **Machine Learning Framework:** PyTorch
- **Configuration:** Pydantic
- **Data Storage:** Zarr
- **Game Interaction:** `py-slippi` and `libmelee`

**Architecture:**

The core of the project is a GPT-based model defined in `model/nano_gpt.py`. This model is trained to predict controller outputs based on game state features.

The project supports two primary training methods:
1.  **Imitation Learning:** The model is trained to mimic human player data from Slippi replay files. The main script for this is `train.py`.
2.  **Reinforcement Learning (PPO):** The model is trained through self-play using Proximal Policy Optimization (PPO). The training process is managed by `train_ppo.py` and can be launched with `run_ppo.sh`. This involves a worker-based system for parallel trajectory collection and an opponent pool for diversity.

The project is highly configurable through `config.py`, which uses Pydantic for structured and validated settings.

## Building and Running

### Installation

1.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Project

**Imitation Learning:**

To train the model using imitation learning, run the `train.py` script:

```bash
python train.py
```

**Reinforcement Learning (PPO):**

To train the model using PPO, you can use the provided shell script. You will need to edit the paths to your Dolphin executable and Melee ISO in the script.

```bash
bash run_ppo.sh
```

Alternatively, you can run `train_ppo.py` directly with the required arguments:

```bash
python train_ppo.py --dolphin-path /path/to/dolphin-emu --iso /path/to/melee.iso
```

**Testing:**

To run the test suite, use `pytest`:

```bash
pytest
```

## Development Conventions

- **Configuration:** All configuration is managed through Pydantic models in `config.py`. This provides a single source of truth for all parameters and ensures that the configuration is well-defined and validated.
- **Typing:** The codebase uses Python's type hints extensively, which helps with code clarity and static analysis.
- **Testing:** Unit tests are located in the `test/` directory and are run using `pytest`.
- **Code Style:** The code generally follows the PEP 8 style guide for Python.
- **Documentation:** The project includes several Markdown files (`README.md`, `QUICK_START.md`, `PPO_IMPLEMENTATION.md`) that provide a good overview of the project and its components.
