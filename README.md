# fast-stein-correction
Implementation of "Boltzmann sampling with quantum annealers via fast Stein correction"

# Set up

```
$ python --version
Python 3.11.0
$ python -m venv .venv
$ source activate .venv/bin
$ pip install -r requirements.txt
```

# Reproducing our experimental results

We provide the source code for the experiments and shell scripts to generate the figure in the paper in the notebook directory.

## Preparation

```
$ cd notebook
$ sh prepare.sh
```

## Run experiments and plot

```
$ sh run_experiments.sh
$ sh mcmc.sh
$ sh plot.sh
```

![](figures/beta_error.png)

