from unittest import TestCase

from parameterized import parameterized

from numpy import asarray, float64
from numpy.testing import assert_almost_equal
from pandas import Series

from finrl.utils import maths


class TestMaths(TestCase):
    @parameterized.expand([
        (
            "one",
            asarray([1, 2, 3]),
            asarray([0.090030573, 0.244728471, 0.665240956])
        ),
        (
            "two",
            asarray([0.1, 0.1, 0.8]),
            asarray([0.249143401, 0.249143401, 0.501713198])
        ),
        (
            "three",
            asarray([0.0, 0.0, 1.0]),
            asarray([0.211941558, 0.211941558, 0.576116885])
        ),
        (
            "Should allow for using a Pandas Series",
            Series([0.0, 0.0, 1.0]),
            asarray([0.211941558, 0.211941558, 0.576116885])
        ),
    ])
    def test_softmax_normalization(self, name, input, expected):
        assert_almost_equal(
            maths.softmax(input),
            expected,
            decimal=9
        )
