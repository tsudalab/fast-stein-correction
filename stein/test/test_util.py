from collections import defaultdict

import numpy as np
import numpy.linalg as LA
import pytest
from dimod.vartypes import Vartype
from numpy.testing import assert_almost_equal

from ..energy import sample_energy
from ..stats import GibbsDistribution
from ..util import (
    IsingSampler,
    gauge_transformation,
    get_all_flipped,
    get_stein_score,
    get_stein_score_continuous,
    onedown,
    onedown_by_h,
    oneup_by_h,
    scale_J_h,
)


class TestOneDown:
    def test_onedown_for_binary(self):
        x = np.array([0, 0, 1])
        actual = onedown(x, 0, Vartype.BINARY)
        expected = np.array([1, 0, 1])
        assert_almost_equal(expected, actual)
        x = np.array([0, 0, 1])
        actual = onedown(x, 2, Vartype.BINARY)
        expected = np.array([0, 0, 0])
        assert_almost_equal(expected, actual)

    def test_oneup_for_spin(self):
        x = np.array([-1, -1, 1])
        actual = onedown(x, 0, Vartype.SPIN)
        expected = np.array([1, -1, 1])
        assert_almost_equal(expected, actual)
        x = np.array([-1, -1, 1])
        actual = onedown(x, 2, Vartype.SPIN)
        expected = np.array([-1, -1, -1])
        assert_almost_equal(expected, actual)

    def test_onedown_for_exception(self):
        x = np.array([0, 0, 1])
        with pytest.raises(
            IndexError, match="Index must be between 0 and 2, but actual was 3"
        ):
            _ = onedown(x, 3, Vartype.BINARY)
        with pytest.raises(
            IndexError, match="Index must be between 0 and 2, but actual was -1"
        ):
            _ = onedown(x, -1, Vartype.BINARY)


class TestOneDownByH:
    def test_onedown_by_h(self):
        x = np.array([0.1, 0, -0.1])
        actual = onedown_by_h(x, 0, 0.000001)
        expected = np.array([0.1 - 0.000001, 0, -0.1])
        assert_almost_equal(expected, actual)
        x = np.array([0.1, 0, -0.1])
        actual = onedown_by_h(x, 2, 0)
        expected = np.array([0.1, 0, -0.1])
        assert_almost_equal(expected, actual)

    def test_onedown_for_exception(self):
        x = np.array([0, 0, 1])
        with pytest.raises(
            IndexError, match="Index must be between 0 and 2, but actual was 3"
        ):
            _ = onedown_by_h(x, 3, 0.1)
        with pytest.raises(
            IndexError, match="Index must be between 0 and 2, but actual was -1"
        ):
            _ = onedown_by_h(x, -1, 0.1)


class TestOneUpByH:
    def test_oneup_by_h(self):
        x = np.array([0.1, 0, -0.1])
        actual = oneup_by_h(x, 0, 0.000001)
        expected = np.array([0.1 + 0.000001, 0, -0.1])
        assert_almost_equal(expected, actual)
        x = np.array([0.1, 0, -0.1])
        actual = oneup_by_h(x, 2, 0)
        expected = np.array([0.1, 0, -0.1])
        assert_almost_equal(expected, actual)
        x = np.array([0.1, 0, -0.1])
        actual = oneup_by_h(x, 2, -0.2)
        expected = np.array([0.1, 0, -0.3])
        assert_almost_equal(expected, actual)

    def test_onedown_for_exception(self):
        x = np.array([0, 0, 1])
        with pytest.raises(
            IndexError, match="Index must be between 0 and 2, but actual was 3"
        ):
            _ = oneup_by_h(x, 3, 0.1)
        with pytest.raises(
            IndexError, match="Index must be between 0 and 2, but actual was -1"
        ):
            _ = oneup_by_h(x, -1, 0.1)


class TestGetAllFlippedData:
    def test_get_all_flipped_binary(self):
        X = np.array([np.array([0, 1, 0, 0]), np.array([0, 0, 0, 1])])
        X_expected = np.array(
            [
                np.array([0, 1, 0, 0]),
                np.array([0, 1, 0, 1]),
                np.array([0, 1, 1, 0]),
                np.array([0, 0, 0, 0]),
                np.array([1, 1, 0, 0]),
                np.array([0, 0, 0, 1]),
                np.array([0, 0, 1, 1]),
                np.array([1, 0, 0, 1]),
            ]
        )
        X_actual = get_all_flipped(X=X, dim=4, vartype=Vartype.BINARY)
        assert len(X_expected) == len(X_actual)
        X_expected_set = set()
        for x in X_expected:
            X_expected_set.add(tuple(x))
        for x in X_actual:
            assert tuple(x) in X_expected_set

    def test_get_all_flipped_spin(self):
        X = np.array(
            [
                np.array([1, -1, -1, -1]),
            ]
        )
        X_expected = np.array(
            [
                np.array([1, -1, -1, -1]),
                np.array([1, -1, -1, 1]),
                np.array([1, -1, 1, -1]),
                np.array([1, 1, -1, -1]),
                np.array([-1, -1, -1, -1]),
            ]
        )
        X_actual = get_all_flipped(X=X, dim=4, vartype=Vartype.SPIN)
        assert len(X_expected) == len(X_actual)
        X_expected_set = set()
        for x in X_expected:
            X_expected_set.add(tuple(x))
        for x in X_actual:
            assert tuple(x) in X_expected_set

    def test_get_all_flipped_empty(self):
        X = np.array([])
        with pytest.raises(ValueError):
            _ = get_all_flipped(X=X, dim=4, vartype=Vartype.SPIN)

    def test_get_all_flipped_single(self):
        X = np.array([-1, -1, 1, -1])
        with pytest.raises(ValueError):
            _ = get_all_flipped(X=X, dim=4, vartype=Vartype.SPIN)


class TestGetSteinScore:
    def test_get_stein_score(self):
        energy = sample_energy()
        mu = GibbsDistribution(energy, 1, 6, compute_pmf=True, vartype=Vartype.BINARY)
        x = np.zeros(6)
        s = get_stein_score(x, mu, Vartype.BINARY)
        tmp1 = 1 - np.exp(-5) / np.exp(-13)
        tmp2 = 1 - np.exp(-8) / np.exp(-13)
        expected = np.array([tmp1, tmp1, tmp1, tmp2, tmp2, tmp2])
        assert_almost_equal(s, expected)


class TestGetSteinScoreContinuous:
    def test_get_stein_score_continuous(self):
        def gaussian(x):
            return np.exp(-LA.norm(x) ** 2 / 2)

        x = np.zeros(6)
        h = 0.01
        s = get_stein_score_continuous(x, gaussian, h)
        x_up = [oneup_by_h(x, i, h) for i in range(len(x))]
        expected = np.array([1 - gaussian(e) / gaussian(x) for e in x_up])
        assert_almost_equal(s, expected)
        h = 1e-5
        s = get_stein_score_continuous(x, gaussian, h)
        x_up = [oneup_by_h(x, i, h) for i in range(len(x))]
        expected = np.array([1 - gaussian(e) / gaussian(x) for e in x_up])
        assert_almost_equal(s, expected)


class Test_scale_J_h:
    def test_valid(self):
        J = {(0, 1): -1, (1, 2): 1, (2, 0): 1}
        h = {0: -1, 1: 1, 2: -1}
        alpha_in = 0.5
        J_new, h_new = scale_J_h(J, h, alpha_in=alpha_in)
        for k in J.keys():
            assert J_new[k] == J[k] * alpha_in
        for k in h.keys():
            assert h_new[k] == h[k] * alpha_in


def gen_test_case(dim, num):
    vectors = []
    for _ in range(num):
        bit_vector = defaultdict(int)
        for i in range(dim):
            bit_vector[i] = np.random.choice([-1, 1])
            vectors.append(bit_vector)
    return vectors


class TestIsingSampler:
    def test_random(self):
        res = IsingSampler.random([0, 1], 4, 4)
        assert len(res) == 4
        assert len(res[0]) == 4
        assert len(list(filter(lambda x: x not in [0, 1], res[0]))) == 0


class TestGaugeTransformation:
    def test_preserverving_value(self):
        h, J = defaultdict(int), defaultdict(int)
        for i in range(8):
            h[i] = np.random.choice([-1, 1])
            for j in range(i + 1, 8):
                J[(i, j)] = np.random.choice([-1, 1])
        vectors = gen_test_case(8, 8)
        for bit_vector in vectors:
            H = 0
            for i in range(8):
                H -= h[i] * bit_vector[i]
                for j in range(i + 1, 8):
                    H -= J[(i, j)] * bit_vector[i] * bit_vector[j]
            h_t, J_t = gauge_transformation(h, J)
            for v in h_t.values():
                assert v in [-1, 1]
            for v in J_t.values():
                assert v in [-1, 1]
            H_t = 0
            for i in range(8):
                H_t -= h_t[i] * bit_vector[i]
                for j in range(i + 1, 8):
                    H_t -= J_t[(i, j)] * bit_vector[i] * bit_vector[j]
            assert H == H_t
