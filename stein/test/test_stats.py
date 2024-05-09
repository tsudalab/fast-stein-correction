import numpy as np
import numpy.testing as npt
import pytest
from dimod.vartypes import Vartype
from numpy import asarray

from ..energy import sample_energy
from ..stats import EmpiricalDistribution, GibbsDistribution


class TestGibbsDistribution:
    def test_init(self):
        energy = sample_energy()
        mu = GibbsDistribution(energy, 1, 6, True, vartype=Vartype.BINARY)
        assert mu._Z > 0

    def test_pmf(self):
        energy = sample_energy()
        mu = GibbsDistribution(energy, 1, 6, True, vartype=Vartype.BINARY)
        total_p = sum([mu.pmf(k) for k in mu.pmf_dict.keys()])
        npt.assert_almost_equal(1.0, total_p, decimal=9)

    def test_pmf_out_of_domain(self):
        energy = sample_energy()
        mu = GibbsDistribution(energy, 1, 6, True, vartype=Vartype.BINARY)
        npt.assert_almost_equal(0.0, mu.pmf([-1, -1, -1, -1, -1, -1]), decimal=9)
        mu = GibbsDistribution(energy, 1, 6, False, vartype=Vartype.BINARY)
        test_vec_invalid = [-1, -1, -1, -1, -1, -1]
        # If partition function is not computed, exp(-H(x)) will be returned.
        npt.assert_almost_equal(
            np.exp(-energy(test_vec_invalid)), mu.pmf(test_vec_invalid)
        )

    def test_pmf_without_partition_factor(self):
        energy = sample_energy()
        mu = GibbsDistribution(energy, 1, 6, False, vartype=Vartype.BINARY)
        test_vec_0 = [0, 0, 0, 0, 0, 0]
        npt.assert_almost_equal(
            np.exp(-energy(test_vec_0)), mu.pmf(test_vec_0), decimal=9
        )
        test_vec_1 = [1, 1, 1, 1, 1, 1]
        test_vec_2 = [1, 1, 0, 0, 1, 1]
        test_vec_list = [test_vec_0, test_vec_1, test_vec_2]
        expected = [np.exp(-energy(t)) for t in test_vec_list]
        actual = mu.pmf(test_vec_list)
        for e, a in zip(expected, actual):
            npt.assert_almost_equal(e, a, decimal=9)


class TestEmpiricalDistribution:
    def test_init(self):
        xk = [[-1, 1], [1, 1]]
        pk = [0.4, 0.6]
        _ = EmpiricalDistribution(xk, pk)
        xk = [[-1], [1]]
        pk = [0.4, 0.6]
        _ = EmpiricalDistribution(xk, pk)
        with pytest.raises(
            ValueError,
            match="Length of samples (.*) does not match length of probs (.*)",
        ):
            _ = EmpiricalDistribution([1], [1.0, 0])

    def test_pmf(self):
        xk = [[-1, 1, 1], [1, 1, -1]]
        pk = [0.4, 0.6]
        emp_dist = EmpiricalDistribution(xk, pk)
        assert emp_dist.pmf(xk[0]) == 0.4
        assert emp_dist.pmf(asarray(xk[0])) == 0.4
        npt.assert_array_almost_equal(emp_dist.pmf(xk), asarray(pk))
        npt.assert_array_almost_equal(emp_dist.pmf(asarray(xk)), asarray(pk))

    def test_rvs(self):
        xk = [[-1, 1], [1, 1], [1, -1]]
        pk = [0.4, 0.3, 0.3]
        emp_dist = EmpiricalDistribution(xk, pk)
        emp_dist.rvs(size=20)
