# pylint: disable=missing-function-docstring, missing-class-docstring
import unittest
import numpy as np
from numpy.testing import assert_array_equal
from tnestmodel.temp_properties import NumberOfTrianglesCalculator

from tnestmodel.temp_fast_graph import SparseTempFastGraph




class TestNumberOfTrianglesCalculator(unittest.TestCase):
    def test_simple1(self,):
        G = SparseTempFastGraph.from_temporal_edges(np.array([(0,1,0), (1,2,1), (1,3,1), (2,0,2), (3,0,3), (2,0,4)],dtype=int), is_directed=True)
        calculator = NumberOfTrianglesCalculator(G)
        calculator.prepare()
        # print(calculator.future_nodes_count, calculator.future_nodes)
        _, arr = G.compute_for_each_slice(calculator.calc_for_slice, min_size=1, call_with_time=True, dtype=int)
        assert_array_equal(arr, [0,3,0,0,0])

    def test_in_time_triangle(self):
        G = SparseTempFastGraph.from_temporal_edges(np.array([(0,1,0), (1,2,0), (2,0,0)],dtype=int), is_directed=True)
        calculator = NumberOfTrianglesCalculator(G, strict=False)
        calculator.prepare()
        # print(calculator.future_nodes_count, calculator.future_nodes)
        _, arr = G.compute_for_each_slice(calculator.calc_for_slice, min_size=1, call_with_time=True, dtype=int)
        assert_array_equal(arr, [3])

        calculator = NumberOfTrianglesCalculator(G, strict=True)
        calculator.prepare()
        # print(calculator.future_nodes_count, calculator.future_nodes)
        _, arr = G.compute_for_each_slice(calculator.calc_for_slice, min_size=1, call_with_time=True, dtype=int)
        assert_array_equal(arr, [0])


if __name__ == '__main__':
    unittest.main(failfast=True)
