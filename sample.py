from stein.energy import create_hamiltonian

dim = 6
import numpy as np
from dimod.vartypes import Vartype

from stein.ksd import boltzmann_correction
from stein.stats import GibbsDistribution
from stein.util import IsingSampler

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
        for j in range(i + 1, len(x)):
            t += J[(i, j)] * x[i] * x[j]
    return t


# H = create_hamiltonian(J, h)

# Input your samples and target inverse temperature $\beta$
num = 150
X_ = IsingSampler.random([-1, 1], dim, num)
X = np.array(list(set([tuple(x) for x in X_])))
beta = 0.5
weights = boltzmann_correction(
    dim=dim,
    samples=X,
    beta=beta,
    hamiltonian=H,
    vartype=Vartype.SPIN,
    eta=1e-2,
    n_iter=5000,
    feature_dim=10000,
    mode="egd"
)
# Compute physical quantity of your interest.

trg = GibbsDistribution(H, beta, dim, True, Vartype.SPIN)
energy_true = 0
for x in trg.pmf_dict.keys():
    energy_true += trg.pmf(x) * H(x)

energy_emp = 0
for x in X_:
    energy_emp += H(x) / len(X_)

# Exact value is approximated by the weights.
energy_stein = 0
for w, x in zip(weights,X):
    # print(w, trg.pmf(x))
    energy_stein += w * H(x)
print("Exact: {}, empirical: {}, Stein correction(egd): {}".format(energy_true, energy_emp, energy_stein))
