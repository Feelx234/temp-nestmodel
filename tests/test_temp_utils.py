# pylint: disable=missing-function-docstring, missing-class-docstring
import unittest
import numpy as np
from numpy.testing import assert_array_equal



from tnestmodel.temp_utils import partition_temporal_edges



class TestTemporalWL(unittest.TestCase):
    def test_partition_temporal_edges_1(self):
        E = np.array([[0,1,0], [0,2,1]],dtype=int)
        res, times = partition_temporal_edges(E)
        assert_array_equal(times, [0,1])
        assert_array_equal(res[0],[[0,1]])
        assert_array_equal(res[1],[[0,2]])

    def test_partition_temporal_edges_2(self):
        E = np.array([[0,1,0],[1,2,0], [0,2,1]],dtype=int)
        res, times = partition_temporal_edges(E)
        assert_array_equal(times, [0,1])
        assert_array_equal(res[0],[[0,1],[1,2]])
        assert_array_equal(res[1],[[0,2]])

    def test_partition_temporal_edges_empty(self):
        E = np.zeros((0,3),dtype=np.int32)
        res, times = partition_temporal_edges(E)
        assert_array_equal(times, [])
        self.assertEqual(len(res),0)

if __name__ == '__main__':
    unittest.main()