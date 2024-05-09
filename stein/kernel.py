from enum import Enum
from typing import Callable, Dict, List, Tuple

import numpy as np
import numpy.linalg as LA
from dimod.vartypes import Vartype

from .stats import AbstractDistribution
from .util import get_all_flipped, get_stein_score, onedown, spin2binary


class KernelType(Enum):
    Gaussian = 1
    Hamming = 2
    Laplace = 3


def hamming_kernel(x: List, xp: List):
    assert len(x) == len(xp)
    diff = [1 if x[i] != xp[i] else 0 for i in range(len(x))]
    return np.exp(-np.sum(diff) / len(x))


def laplace_kernel(x: List, xp: List):
    assert len(x) == len(xp)
    dist = np.abs(x - xp)
    return np.exp(-np.sum(dist))


def gaussian_kernel(x: List, xp: List):
    assert len(x) == len(xp)
    dist = LA.norm(x - xp)
    return np.exp(-(dist**2) / 2)


class FourierBasis:
    """
    Kernel basis expansion based on random feature map.
    Ref:
        Rahimi, Ali, and Benjamin Recht. 2007.
        “Random Features for Large-Scale Kernel Machines.” Advances in Neural Information Processing Systems 20.
        https://proceedings.neurips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html.

    Parameters
    ----------
    domain_dim: int
        Dimensionality of target data.

    feature_dim: int
        Dimensionality of kernel basis.
        Given that E[z(x)^T z(x)] approximates kernel values, dot product of our kernel basis must be the expectation.

    kernel_type: KernelType
        Kernel type you want to use. Hamming, Gaussian, Laplace kernel are supported.

    """

    def __init__(
        self,
        *,
        domain_dim: int,
        feature_dim: int,
        kernel_type: KernelType = KernelType.Gaussian,
    ):
        self._domain_dim = domain_dim
        self._feature_dim = feature_dim
        self._kernel_type = kernel_type
        self._basis_dict: Dict[Tuple[int], List[float]] = {}
        self._params = {}
        self._params["b"] = np.random.uniform(0, 2 * np.pi, size=self._feature_dim)
        if self._kernel_type == KernelType.Gaussian:
            m = np.zeros(self._domain_dim)
            c = np.eye(self._domain_dim, self._domain_dim)
            self._params["w"] = np.random.multivariate_normal(m, c, self._feature_dim)
        elif self._kernel_type == KernelType.Hamming:
            self._params["w"] = (
                np.random.standard_cauchy((self._feature_dim, self._domain_dim))
                / self._domain_dim
            )
        elif self._kernel_type == KernelType.Laplace:
            self._params["w"] = np.random.standard_cauchy(
                (self._feature_dim, self._domain_dim)
            )

    @property
    def get_kernel_type(self):
        return self._kernel_type

    def get_basis(
        self,
        X: List[List],
        vartype: Vartype = Vartype.BINARY,
    ):
        if vartype == Vartype.SPIN and self._kernel_type == KernelType.Hamming:
            X = spin2binary(X)
        return (
            np.sqrt(2)
            * np.cos(X.dot(self._params["w"].T) + self._params["b"])
            / np.sqrt(self._feature_dim)
        )

    # def get_basis(
    #     self,
    #     X: List[int or float] or List[List[int or float]],
    #     vartype: Vartype = Vartype.BINARY,
    # ):
    #     if vartype == Vartype.SPIN and self._kernel_type == KernelType.Hamming:
    #         X = spin2binary(X)
    #     return np.hstack(
    #         (
    #             np.cos(X.dot(self._params["w"].T) + self._params["b"]),
    #             np.sin(X.dot(self._params["w"].T) + self._params["b"]),
    #         )
    #     ) / np.sqrt(self._feature_dim)


class SteinBasis:
    """
    Stein basis expansion.

    Parameters
    ----------
    domain_dim: int
        Dimensionality of target data.

    feature_dim: int
        Dimensionality for fourier basis.
        Final dimension of stein basis becomes feature_dim * domain_dim.

    distrib: AbstractDistribution
        Target distribution of your interest.

    kernel_type: KernelType
        Kernel type you want to use. Hamming, Gaussian, Laplace kernel are supported.

    """

    def __init__(
        self,
        domain_dim: int,
        feature_dim: int,
        *,
        distrib: AbstractDistribution = None,
        func: Callable[[List[float]], float] = None,
        kernel_type: KernelType = KernelType.Gaussian,
        h: float = 1e-5,
    ):
        self._domain_dim = domain_dim
        self._feature_dim = feature_dim
        self._distrib = distrib
        self._func = func
        self._kernel_type = kernel_type
        self._fourier = FourierBasis(
            domain_dim=self._domain_dim,
            feature_dim=self._feature_dim,
            kernel_type=self._kernel_type,
        )
        self._h = h
        self.basis_dict = {}

    def _check(self) -> bool:
        if self._distrib is None and self._func is None:
            raise ValueError

    def get_basis(
        self, X: List[List], vartype: Vartype
    ) -> List[List[float]]:
        if vartype == Vartype.SPIN or vartype == Vartype.BINARY:
            if len(X.shape) == 1 and len(X) > 0:
                X = np.array([X])
            X_all = get_all_flipped(X, dim=self._domain_dim, vartype=vartype)
            fourier_basis = self._fourier.get_basis(X_all, vartype=vartype)
            for x, basis in zip(X_all, fourier_basis):
                key = tuple(x)
                if key not in self.basis_dict.keys():
                    self.basis_dict[key] = basis
            stein_basis = []
            for x in X:
                x_stein_basis = np.array([])
                x_score = get_stein_score(x, self._distrib, vartype=vartype)
                key = tuple(x)
                for i, score in enumerate(x_score):
                    key_onedown = tuple(onedown(x, i, vartype=vartype))
                    v = (1 - score) * self.basis_dict[key] - self.basis_dict[
                        key_onedown
                    ]
                    x_stein_basis = np.append(x_stein_basis, v)
                stein_basis.append(x_stein_basis)
            return np.array(stein_basis)
        else:
            raise NotImplementedError
