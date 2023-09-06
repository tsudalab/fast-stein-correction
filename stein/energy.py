from collections import defaultdict
from typing import Callable, List

import numpy as np
from numpy import asarray


def sample_energy() -> Callable:
    def energy(x: List[int]):
        a1 = np.array([1, 1, 1, 1, 1, 1])
        a2 = np.array([1, 1, 1, 0, 0, 0])
        t1 = 3
        t2 = 2

        ene = (np.dot(x, a1) - t1) ** 2 + (np.dot(x, a2) - t2) ** 2
        return ene

    return energy


def random_energy(dim: int) -> Callable:
    def energy(x: List[int]):
        a1 = np.random.choice([0, 1], dim)
        a2 = np.random.choice([0, 1], dim)
        t1 = dim / 3
        t2 = dim / t1

        ene = (np.dot(x, a1) - t1) ** 2 + (np.dot(x, a2) - t2) ** 2
        return -ene

    return energy


def create_hamiltonian(J: defaultdict, h: defaultdict) -> Callable:
    qubits_candidate = np.array([[e[0], e[1]] for e in J.keys()]).flatten()
    qubits_candidate = np.append(qubits_candidate, np.array(list(h.keys())))
    qubits = sorted(asarray(list(set(qubits_candidate))))

    def hamiltonian(x: List[int]) -> float:
        t1, t2 = 0, 0
        assert len(qubits) == len(x)
        # convert a bit vector into a state vector
        state = defaultdict(int, ((qubits[i], x[i]) for i in range(len(qubits))))
        for i in range(len(qubits)):
            si = qubits[i]
            t2 += h[si] * state[si]
            for j in range(i + 1, len(qubits)):
                sj = qubits[j]
                t1 += J[(si, sj)] * state[sj] * state[si]
        return t1 + t2

    return hamiltonian
