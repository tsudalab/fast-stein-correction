import warnings
from typing import Callable, List, Tuple

import cvxopt
import numpy as np
import numpy.linalg as LA
from cvxopt import solvers
from dimod.vartypes import Vartype

from .exception import NotSupportedError
from .kernel import (
    KernelType,
    SteinBasis,
    gaussian_kernel,
    hamming_kernel,
    laplace_kernel,
)
from .stats import AbstractDistribution, GibbsDistribution
from .util import get_stein_score, onedown

solvers.options["show_progress"] = False


class DiscreteKSD:
    """An implementation of the kernelized Stein discrepancy.

    Yang et al. 2018; Goodness-of-Fit Testing for Discrete Distributions via Stein Discrepancy

    Parameters
    ----------
    dim : int
        Dimension of a sample retrieved in the target space.

    kernel : Callable[[List[int], List[int]], float]
        A kernel function for performing
        the Kernelized Discrete Stein Discrepancy test.

    distrib : AbstractDistribution
        A probability distribution from which we want to approximate
        on the target space.
        It must have a method named pmf by implementing AbstractDistribution class.

    vartype: Vartype
        Denote domain type: SPIN or BINARY.
    """

    def __init__(
        self,
        *,
        dim: int,
        kernel_type: KernelType,
        distrib: AbstractDistribution,
        vartype: Vartype,
    ):
        self._dim = dim
        self._kernel = None
        self._kernel_type = kernel_type
        if self._kernel_type == KernelType.Gaussian:
            self._kernel = gaussian_kernel
        elif self._kernel_type == KernelType.Hamming:
            self._kernel = hamming_kernel
        elif self._kernel_type == KernelType.Laplace:
            self._kernel = laplace_kernel
        self._distrib = distrib
        if vartype != Vartype.BINARY and vartype != Vartype.SPIN:
            raise NotSupportedError
        self._vartype = vartype
        # Fields for optmized results
        self.KP = None
        self.X_stein_basis = None
        self.samples = None
        self.weight = None
        # Fields for memoization
        self.kernel_memo = {}
        self.flip_memo = {}

    def _kernel_memoize(self, x1: List[int], x2: List[int]):
        key1, key2 = (tuple(x1), tuple(x2)), (tuple(x2), tuple(x1))
        if key1 in self.kernel_memo.keys():
            return self.kernel_memo[key1]
        else:
            v = self._kernel(x1, x2)
            self.kernel_memo[key1] = self.kernel_memo[key2] = v
            return v

    def compute_KP(self, X):
        """
        Computes kernelized Stein's identity according to (4) of Black-Box Importance Sampling [Liu and Lee 2017]
        """
        N = len(X)
        tmp = np.zeros([N, N])
        score_list = np.array(
            [get_stein_score(x, self._distrib, self._vartype) for x in X]
        )
        for i in range(N):
            for j in range(i, N):
                x, xp = X[i], X[j]
                score_x = score_list[i]
                score_xp = score_list[j]
                v = 0
                base_kernel = self._kernel_memoize(x, xp)
                for k in range(self._dim):
                    v += base_kernel * (1 - score_x[k]) * (1 - score_xp[k])
                    v -= (1 - score_x[k]) * self._kernel_memoize(
                        x, onedown(xp, k, self._vartype)
                    )
                    v -= (1 - score_xp[k]) * self._kernel_memoize(
                        onedown(x, k, self._vartype), xp
                    )
                    v += self._kernel_memoize(
                        onedown(x, k, self._vartype), onedown(xp, k, self._vartype)
                    )
                tmp[i, j] = tmp[j, i] = v
        return tmp

    def compute_KP_basis(self, X, feature_dim: int = 5000) -> List[List[float]]:
        """
        Computes Stein basis.
        """
        stein_basis = SteinBasis(
            domain_dim=self._dim,
            feature_dim=feature_dim,
            distrib=self._distrib,
            kernel_type=self._kernel_type,
        )
        return stein_basis.get_basis(X, vartype=self._vartype)

    def compute_stein_discrepancy(self, X: List[np.array], w: List[float]):
        if self.KP is None:
            self.KP = self.compute_KP(X)
        return w.dot(self.KP).dot(w)

    def fit(self, X: List[np.array], w: List[float], start: List[float] = None):
        N = len(X)
        assert N == len(w)
        # Compute Kernelized Stein Discrepcy
        if self.KP is None:
            self.KP = self.compute_KP(X)
        # print("!!!!!!!Computed a kernel matrix!!!!!!!")
        # Solve the quadratic linear programming
        # See: https://cvxopt.org/userguide/coneprog.html#quadratic-programming
        P = cvxopt.matrix(self.KP)
        q = cvxopt.matrix(np.zeros(N))
        G = cvxopt.matrix(-np.diag(np.ones(N)))
        h = cvxopt.matrix(np.zeros(N))
        A = cvxopt.matrix(w, (1, N), "d")
        b = cvxopt.matrix(np.array([1.0]), (1, 1))

        if start is None:
            sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        else:
            initvals = {"x": cvxopt.matrix(start, (N, 1), "d")}
            sol = cvxopt.solvers.qp(P, q, G, h, A, b, initvals=initvals)

        # print("!!!!!!!Solved the quadratic programming!!!!!!!")

        # Save the results
        self.samples = X
        self.weight = np.transpose(np.array(sol["x"]))[0] * w

    def fit_egd(
        self,
        X: List[List[int]],
        n_iter: int = 100,
        eta: float = 1e-6,
        feature_dim: int = 5000,
        start: List[float] = None,
        verbose: bool = False,
    ):
        N = len(X)
        w = np.ones(N) / N
        if start is not None:
            w = start
        history = [w]
        if self.X_stein_basis is None:
            self.X_stein_basis = self.compute_KP_basis(X, feature_dim=feature_dim)
        K = self.X_stein_basis.dot(self.X_stein_basis.T)
        self.KP = K
        assert K.shape == (N, N)
        sd = w.dot(K).dot(w)
        X = np.array(X)
        grad = np.zeros(N)
        for it in range(n_iter):
            # flake8: noqa
            grad = (K + K.T).dot(w)
            try:
                with warnings.catch_warnings(record=True) as warn:
                    warnings.simplefilter("always")
                    grad_eta = -eta * grad
                    w_n = w * np.exp(grad_eta)
                    w_n /= sum(w_n)
                    if warn and issubclass(warn[-1].category, RuntimeWarning):
                        raise RuntimeWarning
            except RuntimeWarning:
                grad_eta = -eta * grad
                grad_eta /= np.max(grad_eta)
                w_n = w * np.exp(grad_eta)
                if np.max(w_n) == 0:
                    eta = eta * 0.1
                    break
                w_n /= np.max(w_n)
                w_n /= sum(w_n)
            history.append(w_n)
            tmp = w_n.dot(K).dot(w_n)
            if it % 1000 == 0 and verbose is True:
                print("it: {}, stein discrepancy : {} to {}.".format(it, sd, tmp))
            if tmp < sd:
                sd = tmp
                w = w_n
            else:
                eta *= 0.5
                if eta < 1e-5:
                    break
        self.samples = X
        self.weight = w
        return history

    def get_important_sample_with_probability(self) -> Tuple[List[int], float]:
        idx = np.argmax(self.weight)
        return self.samples[idx], self.weight[idx]


def boltzmann_correction(
    dim: int,
    samples: List[List[int]],
    beta: float,
    hamiltonian: Callable,
    vartype: Vartype,
    kernel_type=KernelType.Hamming,
    n_iter=2000,
    eta=1e-5,
    feature_dim=5000,
    mode="egd",
    start=None,
):
    """Utility function to perform Stein correction about Gibbs-Boltzmann distribution.

    Parameters
    ----------
    dim :
        Dimension of a sample retrieved in the target space.

    samples :
        A kernel function for performing
        the Kernelized Discrete Stein Discrepancy test.

    beta :
        Target inverse temperature of your interest.

    hamiltonian :
        A function representing hamiltonian.
        Takes a binary vector as an input and returns an internal energy.

    vartype :
        Specify dimod.vartypes.Vartype.SPIN or dimod.vartypes.Vartype.BINARY.

    kernel_type :
        Specify one of stein.kernel.KernelType.Hamming or stein.kernel.KernelType.Gaussian or stein.kernel.KernelType.Laplace.

    n_iter :
        Number of iteration for the exponentiated gradient descent.

    eta :
        Learning rate for the exponentiated gradient descent.

    feature_dim :
        Number of random features for approximating base kernel.

    start :
        Initial weights that the egd starts with.
    """
    trg = GibbsDistribution(hamiltonian, beta, dim, False, vartype)
    ksd = DiscreteKSD(
        dim=dim,
        kernel_type=kernel_type,
        distrib=trg,
        vartype=vartype,
    )
    start = np.ones(len(samples)) / len(samples)
    weights = np.copy(start)
    hist = None
    if mode == "egd":
        hist = ksd.fit_egd(
            samples, n_iter=n_iter, eta=eta, feature_dim=feature_dim, start=start
        )
    elif mode == "cvxopt":
        basis = ksd.compute_KP_basis(samples, feature_dim=5000)
        ksd.KP = basis.dot(basis.T)
        ksd.fit(samples, start)
    if ksd.weight is not None:
        weights = ksd.weight
    return weights, ksd.KP, hist
