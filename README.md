# fast-stein-correction
Implementation of "Boltzmann sampling with quantum annealers via fast Stein correction"

## Environment setup

```
$ python --version
Python 3.11.0
$ python -m venv .venv
$ source activate .venv/bin
$ git clone git@github.com:tsudalab/fast-stein-correction.git
$ cd fast-stein-correction
$ pip install -e .
```

## Usage

We show how to apply fast stein correction to your problems.
Whole code is in [`sample.py`](./sample.py).

1. Define your Hamiltonian on binary domain.
   ```python
   dim = 6
   J = {
       (0, 1): -1.0,
       (0, 2): -1.0,
       (0, 3): 1.0,
       (0, 4): -1.0,
       (0, 5): 1.0,
       (1, 2): 1.0,
       (1, 3): -1.0,
       (1, 4): 1.0,
       (1, 5): 1.0,
       (2, 3): -1.0,
       (2, 4): -1.0,
       (2, 5): 1.0,
       (3, 4): 1.0,
       (3, 5): 1.0,
       (4, 5): -1.0,
   }

   h = {
       0: -1.0,
       1: -1.0,
       2: 1.0,
       3: -1.0,
       4: 1.0,
       5: -1.0,
   }

   def H(x):
       assert len(x) == dim
       t = 0
       for i in range(len(x)):
           t += h[i] * x[i]
           for j in range(i+1, len(x)):
               t += J[(i, j)] * x[i] * x[j]
       return t

   ```
   We provide an utility function `create_hamiltonian` to define Ising Hamiltonian given $J$ and $h$.

   ```python
   from stein.energy import create_hamiltonian
   H = create_hamiltonian(J, h)
   ```

2. Input your samples and a target inverse temperature $\beta$
   ```python
   from stein.ksd import boltzmann_correction
   from stein.util import IsingSampler

   from dimod.vartypes import Vartype

   import numpy as np

   num = 150
   X = IsingSampler.random([-1, 1], 6, num)
   X = np.array(list(set([tuple(x) for x in X])))
   beta = 0.5
   weights = boltzmann_correction(
       dim=dim,
       samples=X,
       beta=beta,
       hamiltonian=H,
       vartype=Vartype.SPIN,
       mode="egd",
   )
   ```

3. Compute physical quantity of your interest.
   ```python
   from stein.stats import GibbsDistribution

   # True means a normalization constant for Boltzmann distribution is computed.
   trg = GibbsDistribution(H, beta, dim, True, Vartype.SPIN)
   energy = 0
   for x in trg.pmf_dict.keys():
       energy += trg.pmf(x) * H(x)
   print("Exact", energy) # -> -7.297344197572726
   ```

   Exact value is approximated by the weights.

   ```python
   energy = 0
   for w, x in zip(weights, X):
       energy += w * H(x)
   print("Stein correction:", energy)
   ```

   ```
   $ python sample.py
   Exact: -7.297344197572726, empirical: 0.52, Stein correction(egd): -6.69995997268694
   ```

## Reproducing our experimental results

We provide the source code for the experiments and shell scripts to generate the figure in the ```notebooks/``` directory.

### Preparation

Run scripts to pre-compute exact values for each problem.

```
$ cd notebook
$ sh prepare.sh
```

### Running experiments and plot

```
$ sh run_experiments.sh
$ sh mcmc.sh
$ sh plot.sh
```

![](figures/beta_error.png)

