# pylint: disable=missing-function-docstring, missing-class-docstring
import unittest
import numpy as np
from numpy.testing import assert_array_equal



from tnestmodel.datasets import datasets



class TestTemporalWL(unittest.TestCase):
    def test_smoke_read_datasets_pd(self):
        for dataset in datasets:
            dataset.read_pd()


if __name__ == '__main__':
    unittest.main()
