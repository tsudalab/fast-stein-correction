from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, List

import dimod
import numpy as np
from dimod.vartypes import Vartype
from numpy import asarray
from numpy.random import default_rng


class AbstractDistribution(ABC):
    @abstractmethod
    def pmf(self, k: List[Any]) -> float:
        raise NotImplementedError


class GibbsDistribution(AbstractDistribution):
    def __init__(
        self,
        energy: Callable,
        alpha: float,
        dim: int,
        compute_pmf=False,
        vartype=Vartype.SPIN,
    ):
        self.energy = energy
        self.alpha = alpha
        self.dim = dim
        self._pmf_called = False
        self._Z = 0
        self.pmf_dict = defaultdict(float)
        self.compute_pmf = False
        self.vartype = vartype
        if compute_pmf is True:
            self.compute_probability_mass_function()
            self.compute_pmf = True

    def pmf(self, k: List[Any], loc=0):
        sh = asarray(k).shape
        if len(sh) == 1:
            if not self.compute_pmf:
                return np.exp(-self.alpha * self.energy(k))
            else:
                return self.pmf_dict[tuple(k)]
        elif len(sh) == 2:
            if not self.compute_pmf:
                return asarray([np.exp(-self.alpha * self.energy(s)) for s in k])
            else:
                return asarray([self.pmf_dict[tuple(x)] for x in k])

    def compute_probability_mass_function(self) -> None:
        for n in range(2**self.dim):
            b = np.binary_repr(n, width=self.dim)
            sigma = np.array([])
            if self.vartype is Vartype.BINARY:
                sigma = np.array(list(map(lambda x: int(x), list(b))))
            elif self.vartype is Vartype.SPIN:
                sigma = np.array(list(map(lambda x: 2 * int(x) - 1, list(b))))
            tmp = np.exp(-self.alpha * self.energy(sigma))
            self.pmf_dict[tuple(sigma)] = tmp
            self._Z += tmp

        for k in self.pmf_dict.keys():
            self.pmf_dict[k] /= self._Z

        self.compute_pmf = True


class EmpiricalDistribution(AbstractDistribution):
    def __init__(self, xk: List[List[Any]], pk: List[float]):
        self.xk = xk
        self.pk = pk
        if len(xk) != len(pk):
            raise ValueError(
                f"Length of samples ({len(xk)}) does not match length of probs ({len(pk)})"
            )
        self.pmf_dict = defaultdict(float)
        for i in range(len(xk)):
            self.pmf_dict[tuple(xk[i])] = pk[i]

    def pmf(self, k: List[Any], loc=0):
        sh = asarray(k).shape
        if len(sh) == 1:
            return self.pmf_dict[tuple(k)]
        elif len(sh) == 2:
            return asarray([self.pmf_dict[tuple(x)] for x in k])

    def rvs(self, size=1, replace=True):
        rng = default_rng()
        return rng.choice(self.xk, size, replace=replace, p=self.pk)


def create_empirical_distribution(
    sampleset: dimod.SampleSet,
) -> EmpiricalDistribution:
    # record looks like [([-1, -1], -1., 92, 0.) ([ 1, -1], -1., 23, 0.) ([ 1, 1], -1., 85, 0.)]
    record = sampleset.record
    pk = record["num_occurrences"] / sum(record["num_occurrences"])
    xk = record["sample"]
    return EmpiricalDistribution(xk, pk)
