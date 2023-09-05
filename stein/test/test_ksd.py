import numpy as np
import numpy.linalg as LA
from dimod.vartypes import Vartype
from numpy.testing import assert_almost_equal

from ..energy import sample_energy
from ..kernel import KernelType
from ..ksd import DiscreteKSD
from ..stats import EmpiricalDistribution, GibbsDistribution


def int2bin(i, dim):
    y = format(i, f"0{dim}b")
    # z denotes a state: e.x. [0, 1, 0, 0, 1, 0]
    z = np.zeros(dim, dtype="i1")
    for j in range(dim):
        z[j] = int(y[j])
    return z


class TestDiscreteKSD:
    def test_exact_and_stein_basis_approx(self):
        """
        Check if Stein basis X approximates kernelized Stein discrepancy K by comparing the exact kernel K
          and reconstracted X'X.
        """
        energy = sample_energy()
        mu = GibbsDistribution(energy, 1, 6, compute_pmf=True, vartype=Vartype.BINARY)
        ksd = DiscreteKSD(
            dim=6,
            distrib=mu,
            kernel_type=KernelType.Hamming,
            vartype=Vartype.BINARY,
        )
        samples = np.array(
            [
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 1, 1],
                [1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ]
        )
        K_exact = ksd.compute_KP(samples)
        X_basis = ksd.compute_KP_basis(samples, feature_dim=5000)
        K_approx = X_basis.dot(X_basis.T)
        # Normalize for testing
        K_exact /= LA.norm(K_exact)
        K_approx /= LA.norm(K_approx)
        print(K_exact)
        print(K_approx)
        assert LA.norm(K_exact - K_approx) < 0.05

    def test_fit(self):
        """
        Check if the resulting weight vector sum up to 1.0.
        """
        energy = sample_energy()
        mu = GibbsDistribution(energy, 1, 6, compute_pmf=True, vartype=Vartype.BINARY)
        ksd = DiscreteKSD(
            dim=6,
            distrib=mu,
            kernel_type=KernelType.Hamming,
            vartype=Vartype.BINARY,
        )
        samples = np.array([[0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 1, 1], [1, 0, 0, 0, 0, 0]])
        ksd.fit(samples, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
        assert_almost_equal(1, sum(ksd.weight))

    def test_fit_accuracy(self):
        # Problem setup
        dim = 6
        ub = 2**dim
        beta = 1  # target temperature
        energy = sample_energy()
        ht = GibbsDistribution(
            energy, beta, dim, compute_pmf=True, vartype=Vartype.BINARY
        )
        # Define ksd and related components
        target = GibbsDistribution(
            energy, beta, dim, compute_pmf=False, vartype=Vartype.BINARY
        )
        X = []
        for i in range(ub):
            ztmp = int2bin(i, dim)
            X.append(ztmp)
        ksd = DiscreteKSD(
            dim=dim,
            kernel_type=KernelType.Gaussian,
            distrib=target,
            vartype=Vartype.BINARY,
        )
        w = np.ones(len(X))
        ksd.fit(X, w)
        corr = EmpiricalDistribution(X, ksd.weight)
        for x in X:
            print(x, corr.pmf(x), ht.pmf(x))
        for x in X:
            assert_almost_equal(corr.pmf(x), ht.pmf(x), decimal=3)

    def test_fit_egd(self):
        dim = 6
        ub = 2**dim
        beta = 1  # target temperature
        energy = sample_energy()
        ht = GibbsDistribution(
            energy, beta, dim, compute_pmf=True, vartype=Vartype.BINARY
        )
        # Define ksd and related components
        target = GibbsDistribution(
            energy, beta, dim, compute_pmf=False, vartype=Vartype.BINARY
        )
        X = []
        for i in range(ub):
            ztmp = int2bin(i, dim)
            X.append(ztmp)
        ksd = DiscreteKSD(
            dim=dim,
            kernel_type=KernelType.Gaussian,
            distrib=target,
            vartype=Vartype.BINARY,
        )
        X = np.array(X)
        ksd.fit_egd(X, feature_dim=5000, n_iter=5000, eta=0.01)
        corr = EmpiricalDistribution(X, ksd.weight)
        for x in X:
            print(x, corr.pmf(x), ht.pmf(x))
        for x in X:
            assert (corr.pmf(x) - ht.pmf(x)) ** 2 < 1e-6
