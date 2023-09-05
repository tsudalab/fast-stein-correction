from typing import Callable, List

import numpy as np
import pytest
from dimod.vartypes import Vartype
from numpy.testing import assert_almost_equal

from ..kernel import (
    FourierBasis,
    KernelType,
    SteinBasis,
    gaussian_kernel,
    hamming_kernel,
    laplace_kernel,
)
from ..stats import GibbsDistribution
from ..util import get_stein_score, onedown


class TestGaussianKernel:
    def test_gaussian_kernel(self):
        x = np.array([1, 1, 0])
        xp = np.array([1, 1, 1])
        assert_almost_equal(np.exp(-1 / 2), gaussian_kernel(x, xp), decimal=6)
        x = np.array([1, 1, -1])
        xp = np.array([1, 1, 1])
        assert_almost_equal(np.exp(-4 / 2), gaussian_kernel(x, xp), decimal=6)

    def test_gaussian_kernel_assertion(self):
        x = np.array([1, 1, 0, 1])
        xp = np.array([1, 1, 1])
        with pytest.raises(AssertionError):
            _ = gaussian_kernel(x, xp)


class TestHammingKernel:
    def test_hamming_kernel(self):
        x = np.array([1, 1, 0])
        xp = np.array([1, 1, 1])
        assert_almost_equal(np.exp(-1 / 3), hamming_kernel(x, xp), decimal=6)
        x = np.array([1, 1, -1])
        xp = np.array([1, -1, 1])
        assert_almost_equal(np.exp(-2 / 3), hamming_kernel(x, xp), decimal=6)

    def test_hamming_kernel_assertion(self):
        x = np.array([1, 1, 0, 1])
        xp = np.array([1, 1, 1])
        with pytest.raises(AssertionError):
            _ = hamming_kernel(x, xp)


class TestLaplaceKernel:
    def test_laplace_kernel(self):
        x = np.array([1, 1, 0])
        xp = np.array([1, 1, 1])
        assert_almost_equal(np.exp(-1), laplace_kernel(x, xp), decimal=6)
        x = np.array([1, 1, -1])
        xp = np.array([1, -1, 1])
        assert_almost_equal(np.exp(-4), laplace_kernel(x, xp), decimal=6)

    def test_laplace_kernel_assertion(self):
        x = np.array([1, 1, 0, 1])
        xp = np.array([1, 1, 1])
        with pytest.raises(AssertionError):
            _ = laplace_kernel(x, xp)


class TestFourierBasis:
    # Batch computation is supported.
    def test_get_basis_gaussian(self):
        X = np.array(
            [
                np.array([1, 0, 0]),
                np.array([1, 1, 0]),
                np.array([1, 0, 1]),
                np.array([0, 1, 1]),
            ]
        )
        fb = FourierBasis(
            domain_dim=3, feature_dim=3000, kernel_type=KernelType.Gaussian
        )
        X_basis = fb.get_basis(X)
        for i in range(len(X)):
            for j in range(i, len(X)):
                assert_almost_equal(
                    gaussian_kernel(X[i], X[j]), X_basis[i].dot(X_basis[j]), decimal=1
                )

    # Single input is also supported
    def test_get_basis_gaussian_with_single_data(self):
        dim = 5
        x = np.array([1, 0, 0, 0, 1])
        xp = np.array([1, 1, 1, 0, 0])
        fb = FourierBasis(
            domain_dim=dim, feature_dim=3000, kernel_type=KernelType.Gaussian
        )
        x_basis = fb.get_basis(x)
        xp_basis = fb.get_basis(xp)
        assert_almost_equal(gaussian_kernel(x, xp), x_basis.dot(xp_basis), decimal=1)

    def test_get_basis_gaussian_real(self):
        X = np.array(
            [
                np.array([1.1, 0.2, 0.6]),
                np.array([-0.1, 0.4, 0.2]),
                np.array([0.7, -1.4, -2]),
            ]
        )
        fb = FourierBasis(
            domain_dim=3, feature_dim=3000, kernel_type=KernelType.Gaussian
        )
        X_basis = fb.get_basis(X, vartype=Vartype.REAL)
        for i in range(len(X)):
            for j in range(i, len(X)):
                assert_almost_equal(
                    gaussian_kernel(X[i], X[j]), X_basis[i].dot(X_basis[j]), decimal=1
                )

    def test_get_basis_gaussian_with_real_data(self):
        dim = 5
        x = np.array([0.1, 0, 0, 0, 0.5])
        xp = np.array([1, -0.1, 0.1, 0, 0])
        fb = FourierBasis(
            domain_dim=dim, feature_dim=3000, kernel_type=KernelType.Gaussian
        )
        x_basis = fb.get_basis(x)
        xp_basis = fb.get_basis(xp)
        assert_almost_equal(gaussian_kernel(x, xp), x_basis.dot(xp_basis), decimal=1)

    # Hamming kernel, multiple input
    def test_get_basis_hamming(self):
        dim = 3
        X = np.array(
            [
                np.array([1, 0, 0]),
                np.array([1, 1, 0]),
                np.array([1, 0, 1]),
                np.array([0, 1, 1]),
            ]
        )
        fb = FourierBasis(
            domain_dim=dim, feature_dim=3000, kernel_type=KernelType.Hamming
        )
        X_basis = fb.get_basis(X)
        for i in range(len(X)):
            for j in range(i, len(X)):
                assert_almost_equal(
                    hamming_kernel(X[i], X[j]),
                    X_basis[i].dot(X_basis[j]),
                    decimal=1,
                )

    # Hamming kernel, multiple input, vartype=SPIN
    def test_get_basis_hamming_spin(self):
        dim = 3
        X = np.array(
            [
                np.array([1, -1, -1]),
                np.array([1, 1, -1]),
                np.array([1, -1, 1]),
                np.array([-1, 1, 1]),
            ]
        )
        fb = FourierBasis(
            domain_dim=dim, feature_dim=3000, kernel_type=KernelType.Hamming
        )
        X_basis = fb.get_basis(X, vartype=Vartype.SPIN)
        for i in range(len(X)):
            for j in range(i, len(X)):
                assert_almost_equal(
                    hamming_kernel(X[i], X[j]),
                    X_basis[i].dot(X_basis[j]),
                    decimal=1,
                )

    # Hamming kernel, single input
    def test_get_basis_hamming_with_single_data(self):
        dim = 5
        x = np.array([1, 0, 0, 0, 1])
        xp = np.array([1, 1, 1, 0, 0])
        fb = FourierBasis(
            domain_dim=dim, feature_dim=3000, kernel_type=KernelType.Hamming
        )
        x_basis = fb.get_basis(x)
        xp_basis = fb.get_basis(xp)
        assert_almost_equal(hamming_kernel(x, xp), x_basis.dot(xp_basis), decimal=1)

    # Laplace kernel, multiple input
    def test_get_basis_laplace(self):
        dim = 3
        X = np.array(
            [
                np.array([1, 0, 0]),
                np.array([1, 1, 0]),
                np.array([1, 0, 1]),
                np.array([0, 1, 1]),
            ]
        )
        fb = FourierBasis(
            domain_dim=dim, feature_dim=3000, kernel_type=KernelType.Laplace
        )
        X_basis = fb.get_basis(X)
        for i in range(len(X)):
            for j in range(i, len(X)):
                assert_almost_equal(
                    laplace_kernel(X[i], X[j]),
                    X_basis[i].dot(X_basis[j]),
                    decimal=1,
                )

    # Laplace kernel, multiple input
    def test_get_basis_laplace_real(self):
        dim = 3
        X = np.array(
            [
                np.array([1.1, 0.2, 0.6]),
                np.array([-0.1, 0.4, 0.2]),
                np.array([0.7, -1.4, -2]),
            ]
        )
        fb = FourierBasis(
            domain_dim=dim, feature_dim=3000, kernel_type=KernelType.Laplace
        )
        X_basis = fb.get_basis(X, vartype=Vartype.REAL)
        for i in range(len(X)):
            for j in range(i, len(X)):
                assert_almost_equal(
                    laplace_kernel(X[i], X[j]),
                    X_basis[i].dot(X_basis[j]),
                    decimal=1,
                )

    # Laplace kernel, single input
    def test_get_basis_laplace_with_single_data(self):
        dim = 5
        x = np.array([1, 0, 0, 0, 1])
        xp = np.array([1, 1, 1, 0, 0])
        fb = FourierBasis(
            domain_dim=dim, feature_dim=3000, kernel_type=KernelType.Laplace
        )
        x_basis = fb.get_basis(x)
        xp_basis = fb.get_basis(xp)
        assert_almost_equal(laplace_kernel(x, xp), x_basis.dot(xp_basis), decimal=1)

    def test_get_basis_laplace_with_real_data(self):
        dim = 5
        x = np.array([0.1, 0, 0, 0, 0.5])
        xp = np.array([1, -0.1, 0.1, 0, 0])
        fb = FourierBasis(
            domain_dim=dim, feature_dim=3000, kernel_type=KernelType.Laplace
        )
        x_basis = fb.get_basis(x, vartype=Vartype.REAL)
        xp_basis = fb.get_basis(xp, vartype=Vartype.REAL)
        assert_almost_equal(laplace_kernel(x, xp), x_basis.dot(xp_basis), decimal=1)


def stein_kernel(
    x: List[int],
    xp: List[int],
    sx: List[float],
    sxp: List[float],
    vartype: Vartype,
    kernel: Callable[[List[int]], float],
) -> float:
    v = 0
    domain_dim = len(x)
    base_kernel = kernel(x, xp)
    for k in range(domain_dim):
        v += base_kernel * (1 - sx[k]) * (1 - sxp[k])
        v -= (1 - sx[k]) * kernel(x, onedown(xp, k, vartype))
        v -= (1 - sxp[k]) * kernel(onedown(x, k, vartype), xp)
        v += kernel(onedown(x, k, vartype), onedown(xp, k, vartype))
    return v


class TestSteinBasis:
    def test_get_basis_single(self):
        def hamiltonian(x):
            w = np.array([-1, -1, 1])
            return w.dot(x)

        X = np.array([1, 0, 1])
        domain_dim = 3
        feature_dim = 100
        vartype = Vartype.BINARY
        kernel_type = KernelType.Gaussian
        distrib = GibbsDistribution(hamiltonian, 0.1, domain_dim, False, vartype)
        stein_basis = SteinBasis(
            domain_dim=domain_dim,
            feature_dim=feature_dim,
            distrib=distrib,
            kernel_type=kernel_type,
        )
        X_basis = stein_basis.get_basis(X, vartype=vartype)
        assert len(X_basis[0]) == domain_dim * feature_dim

    def test_get_basis_hamming_binary(self):
        def hamiltonian(x):
            w = np.array([-1, -1, 1, -1])
            return w.dot(x)

        X = np.array(
            [
                np.array([1, 0, 1, 0]),
                np.array([0, 1, 0, 0]),
                np.array([1, 1, 0, 1]),
            ]
        )
        domain_dim = 4
        feature_dim = 10000
        vartype = Vartype.BINARY
        kernel_type = KernelType.Hamming
        kernel = hamming_kernel
        distrib = GibbsDistribution(hamiltonian, 0.1, domain_dim, False, vartype)
        stein_basis = SteinBasis(
            domain_dim=domain_dim,
            feature_dim=feature_dim,
            distrib=distrib,
            kernel_type=kernel_type,
        )
        X_basis = stein_basis.get_basis(X, vartype=vartype)
        score_list = np.array([get_stein_score(x, distrib, vartype) for x in X])
        for i in range(len(X)):
            for j in range(i, len(X)):
                x, xp = X[i], X[j]
                score_x = score_list[i]
                score_xp = score_list[j]
                v = stein_kernel(x, xp, score_x, score_xp, vartype, kernel)
                print(x, xp)
                assert_almost_equal(X_basis[i].dot(X_basis[j]), v, decimal=1)

    def test_get_basis_gaussian_binary(self):
        def hamiltonian(x):
            w = np.array([-1, -1, 1, -1])
            return w.dot(x)

        X = np.array(
            [
                np.array([1, 0, 1, 0]),
                np.array([0, 1, 0, 0]),
                np.array([1, 1, 0, 1]),
            ]
        )
        domain_dim = 4
        feature_dim = 10000
        vartype = Vartype.BINARY
        kernel_type = KernelType.Gaussian
        kernel = gaussian_kernel
        distrib = GibbsDistribution(hamiltonian, 0.1, domain_dim, False, vartype)
        stein_basis = SteinBasis(
            domain_dim=domain_dim,
            feature_dim=feature_dim,
            distrib=distrib,
            kernel_type=kernel_type,
        )
        X_basis = stein_basis.get_basis(X, vartype=vartype)
        score_list = np.array([get_stein_score(x, distrib, vartype) for x in X])
        for i in range(len(X)):
            for j in range(i, len(X)):
                x, xp = X[i], X[j]
                score_x = score_list[i]
                score_xp = score_list[j]
                v = stein_kernel(x, xp, score_x, score_xp, vartype, kernel)
                print(x, xp)
                assert_almost_equal(X_basis[i].dot(X_basis[j]), v, decimal=1)

    def test_get_basis_laplace_binary(self):
        def hamiltonian(x):
            w = np.array([-1, -1, 1, -1])
            return w.dot(x)

        X = np.array(
            [
                np.array([1, 0, 1, 0]),
                np.array([0, 1, 0, 0]),
                np.array([1, 1, 0, 1]),
            ]
        )
        domain_dim = 4
        feature_dim = 10000
        vartype = Vartype.BINARY
        kernel_type = KernelType.Laplace
        kernel = laplace_kernel
        distrib = GibbsDistribution(hamiltonian, 0.1, domain_dim, False, vartype)
        stein_basis = SteinBasis(
            domain_dim=domain_dim,
            feature_dim=feature_dim,
            distrib=distrib,
            kernel_type=kernel_type,
        )
        X_basis = stein_basis.get_basis(X, vartype=vartype)
        score_list = np.array([get_stein_score(x, distrib, vartype) for x in X])
        for i in range(len(X)):
            for j in range(i, len(X)):
                x, xp = X[i], X[j]
                score_x = score_list[i]
                score_xp = score_list[j]
                v = stein_kernel(x, xp, score_x, score_xp, vartype, kernel)
                print(x, xp)
                assert_almost_equal(X_basis[i].dot(X_basis[j]), v, decimal=1)

    def test_get_basis_hamming_spin(self):
        def hamiltonian(x):
            w = np.array([-1, -1, 1, -1])
            return w.dot(x)

        X = np.array(
            [
                np.array([1, -1, 1, -1]),
                np.array([-1, 1, -1, -1]),
                np.array([1, 1, -1, 1]),
            ]
        )
        domain_dim = 4
        feature_dim = 10000
        vartype = Vartype.SPIN
        kernel_type = KernelType.Hamming
        kernel = hamming_kernel
        distrib = GibbsDistribution(hamiltonian, 0.1, domain_dim, False, vartype)
        stein_basis = SteinBasis(
            domain_dim=domain_dim,
            feature_dim=feature_dim,
            distrib=distrib,
            kernel_type=kernel_type,
        )
        X_basis = stein_basis.get_basis(X, vartype=vartype)
        score_list = np.array([get_stein_score(x, distrib, vartype) for x in X])
        for i in range(len(X)):
            for j in range(i, len(X)):
                x, xp = X[i], X[j]
                score_x = score_list[i]
                score_xp = score_list[j]
                v = stein_kernel(x, xp, score_x, score_xp, vartype, kernel)
                print(x, xp)
                assert_almost_equal(X_basis[i].dot(X_basis[j]), v, decimal=1)

    def test_get_basis_gaussian_spin(self):
        def hamiltonian(x):
            w = np.array([-1, -1, 1, -1])
            return w.dot(x)

        X = np.array(
            [
                np.array([1, -1, 1, -1]),
                np.array([-1, 1, -1, -1]),
                np.array([1, 1, -1, 1]),
            ]
        )
        domain_dim = 4
        feature_dim = 10000
        vartype = Vartype.SPIN
        kernel_type = KernelType.Gaussian
        kernel = gaussian_kernel
        distrib = GibbsDistribution(hamiltonian, 0.1, domain_dim, False, vartype)
        stein_basis = SteinBasis(
            domain_dim=domain_dim,
            feature_dim=feature_dim,
            distrib=distrib,
            kernel_type=kernel_type,
        )
        X_basis = stein_basis.get_basis(X, vartype=vartype)
        score_list = np.array([get_stein_score(x, distrib, vartype) for x in X])
        for i in range(len(X)):
            for j in range(i, len(X)):
                x, xp = X[i], X[j]
                score_x = score_list[i]
                score_xp = score_list[j]
                v = stein_kernel(x, xp, score_x, score_xp, vartype, kernel)
                print(x, xp)
                assert_almost_equal(X_basis[i].dot(X_basis[j]), v, decimal=1)

    def test_get_basis_laplace_spin(self):
        def hamiltonian(x):
            w = np.array([-1, -1, 1, -1])
            return w.dot(x)

        X = np.array(
            [
                np.array([1, -1, 1, -1]),
                np.array([-1, 1, -1, -1]),
                np.array([1, 1, -1, 1]),
            ]
        )
        domain_dim = 4
        feature_dim = 10000
        vartype = Vartype.SPIN
        kernel_type = KernelType.Laplace
        kernel = laplace_kernel
        distrib = GibbsDistribution(hamiltonian, 0.1, domain_dim, False, vartype)
        stein_basis = SteinBasis(
            domain_dim=domain_dim,
            feature_dim=feature_dim,
            distrib=distrib,
            kernel_type=kernel_type,
        )
        X_basis = stein_basis.get_basis(X, vartype=vartype)
        score_list = np.array([get_stein_score(x, distrib, vartype) for x in X])
        for i in range(len(X)):
            for j in range(i, len(X)):
                x, xp = X[i], X[j]
                score_x = score_list[i]
                score_xp = score_list[j]
                v = stein_kernel(x, xp, score_x, score_xp, vartype, kernel)
                print(x, xp)
                assert_almost_equal(X_basis[i].dot(X_basis[j]), v, decimal=1)
