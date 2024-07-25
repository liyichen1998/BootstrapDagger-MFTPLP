About Me: Yichen Li www.linkedin.com/in/yichen-li-14004a20b

# BootstrapDagger-MFTPLP

ICML 2024：Agnostic Interactive Imitation Learning: New Theory and Practical Algorithms

Paper link: https://proceedings.mlr.press/v235/li24ck.html  arXiv: https://arxiv.org/abs/2312.16860

BootstrapDagger-MFTPLP is an advanced implementation built upon the Disagreement-Regularized Imitation Learning (DRIL) framework https://github.com/xkianteb/dril. It features a new online learning pipeline with support for parallelized ensemble policies. The repository includes implementations of `DAgger`, `MFTPL-P`, and `Bootstrap-DAgger`.

## Features

- **Ensemble Learning**: Implements parallelized ensemble policies.
- **Task Environments**: Tested on continuous control tasks `HalfCheetahBulletEnv-v0`, `AntBulletEnv-v0`, `Walker2DBulletEnv-v0`, and `HopperBulletEnv-v0`.
- **Comprehensive Experiments**: Includes hyperparameter tuning, performance evaluation, and comparisons with baseline algorithms.

## Setup

### Experiment environment:

- Ubuntu with a 3.3 GHz Intel Core i9 CPU
- 4 NVIDIA GeForce RTX 2080 Ti GPUs
- Python 3.x
- Required Python libraries: "stable-baselines", "rl-baselines-zoo", "baselines", "gym", "pytorch", "pybullet"

### Installation

To set up the project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/liyichen1998/BootstrapDagger-MFTPLP.git
    cd BootstrapDagger-MFTPLP
    ```

2. Install the required dependencies

## Usage

### Running Experiments

To run experiments, use the provided Python scripts. Here is an example command:

```bash
python run_experiment.py --algorithm <algorithm_name> --task <task_name>
