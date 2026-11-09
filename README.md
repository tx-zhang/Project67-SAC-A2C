# Project 67 SAC and A2C

This repository contains the SAC and recurrent A2C experiments for the Jetbot navigation task. The shared navigation environment uses LiDAR, relative goal information, continuous velocity commands and a fixed obstacle layout.

## Environment

The SAC and A2C experiments used Python 3.10, Isaac Sim 4.5.0, Isaac Lab 2.1.1 and PyTorch 2.7.0. The PPO implementation uses a separate simulator setup with the same navigation task definition. The scene assets created for this project are in `assets/`.

Install Isaac Sim and Isaac Lab in the active Python environment. The package versions used by these experiments are listed in `constraints-isaac211.txt`. From the repository root, install the project and the A2C dependencies:

```powershell
python -m pip install -e ".[a2c]" -c constraints-isaac211.txt
```

## SAC

```powershell
python -m scripts.sac_experiment --output runs/sac/gamma-0999
python -m scripts.sac --backend lab211 --mode eval --headless --device cuda:0 --num_envs 1 --episodes 3 --checkpoint models/sac-400000.pt
```

The supplied checkpoint is from 400,000 transitions. Its three deterministic evaluation episodes ended in timeout, with a mean return of -20.49.

## Recurrent A2C with gSDE

```powershell
python -m scripts.a2c --headless --device cuda:0 --num_envs 32 --gsde --rollouts 128 --sequence_length 128 --noise_period 30 --output runs/a2c/gsde32
python -m scripts.a2c --mode eval --headless --device cuda:0 --num_envs 1 --episodes 3 --gsde --rollouts 128 --sequence_length 128 --noise_period 30 --checkpoint models/a2c-gsde-final.pt --output runs/a2c/eval
```

The supplied final training checkpoint records 14,348,288 cumulative transitions. The final training session recorded zero successes. The A2C plots describe training observations.

`results/` contains the recorded SAC evaluations and trajectory, A2C training and motion plots, and their exported data.
