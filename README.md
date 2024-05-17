## WFCRL Algorithms

This repository contains the source code for the WFCRL multi-agent RL benchmark.
The benchmark is done on the [WFCRL environment suite](https://github.com/ifpen/wfcrl-env).

All experiments are adapted from the [CleanRL](https://github.com/vwxyzjn/cleanrl) repository.
Algorithms:


| **Algorithm**        | **File** | **Description**     |
|----------------------------------|--------------------|--------------------------------------------------------------------------------------|
| IPPO           | `algos/baseline_ippo.py`   | See [Yu et. al](https://arxiv.org/abs/2103.01955)            |
| MAPPO          | `algos/baseline_mappo.py`  |  See [Yu et. al](https://arxiv.org/abs/2103.01955)     |
| IFAC           | `algos/ifac.py`     | Simple online actor critic with Fourier Basis     |

Scripts with the `windrose` suffix train under *Wind Scenario II*. Other implement *Wind Snecario I*.

Install the dependencies:
```
pip install -r requirements
```

Launch an IPPO training experiment on the `Dec_Ablaincourt_Floris` environment:

```
python algos/baseline_ippo.py --seed 1 --env_id Dec_Ablaincourt_Floris --total_timesteps 1000000
```

Evaluate it on the on the `Dec_Ablaincourt_Fastfarm` environment:

```
mpiexec -n 1 python algos/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --total_timesteps 10000 --pretrained_models path/to/run
```

Experiments for training and evaluation runs are in the `scripts` folder.


To track the experiment in Wandb, add your API key in an `.env` file at the root of the folder:

```
WANDB_API_KEY=you_api_key
```