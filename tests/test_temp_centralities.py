# pylint: disable=missing-function-docstring, missing-class-docstring
import unittest
import numpy as np
#from numpy.testing import assert_array_equal
from tnestmodel.temp_fast_graph import TempFastGraph, SparseTempFastGraph
from tnestmodel.temp_centralities import calc_temp_katz, calc_temp_katz_from_causal, calc_temp_katz_iter


edges0 = np.array([[2,1]], dtype=np.int32)
edges1 = np.array([[1,2]], dtype=np.int32)
edges2 = np.array([[2,3]], dtype=np.int32)
edges3 = np.array([[0,1],[1,2], [2,0]], dtype=np.int32)
temp_edges1 = [edges0.copy(), edges1.copy(), edges2, edges3, edges0]
solutions1 = [[1.11211211, 1.25224224, 1.43643544, 1.        ], [1.12312312, 1.33543544, 1.23123123, 1.111     ]]

kinds = ("broadcast", "receive")

class TestTCentralities(unittest.TestCase):
    def test_sparse_katz(self):
        G = SparseTempFastGraph(temp_edges1, is_directed=True)
        for kind, solution in zip(kinds, solutions1):
            np.testing.assert_almost_equal(solution, calc_temp_katz(G, kind=kind))

    def test_sparse_katz_causal(self):
        G = SparseTempFastGraph(temp_edges1, is_directed=True)
        for kind, solution in zip(kinds, solutions1):
            np.testing.assert_almost_equal(solution, calc_temp_katz_from_causal(G, kind=kind))

    def test_sparse_katz_iter(self):
        G = SparseTempFastGraph(temp_edges1, is_directed=True)
        for kind, solution in zip(kinds, solutions1):
            with self.subTest(kind=kind):
                np.testing.assert_almost_equal(solution, calc_temp_katz_iter(G, kind=kind))

    def test_katz(self):
        G = TempFastGraph(temp_edges1, is_directed=True)
        for kind, solution in zip(kinds, solutions1):
            np.testing.assert_almost_equal(solution, calc_temp_katz(G, kind=kind))

    def test_katz_causal(self):
        G = TempFastGraph(temp_edges1, is_directed=True)
        for kind, solution in zip(kinds, solutions1):
            np.testing.assert_almost_equal(solution, calc_temp_katz_from_causal(G, kind=kind))

if __name__ == '__main__':
    unittest.main()
