# pylint: disable=missing-function-docstring, missing-class-docstring
import unittest
import numpy as np
from numpy.testing import assert_array_equal
#from tnestmodel.temp_fast_graph import TempFastGraph, SparseTempFastGraph
#from tnestmodel.temp_fast_graph import get_rolling_max_degree


from tnestmodel.temp_wl import compute_d_rounds, _compute_d_rounds, TemporalColorsStruct
from tnestmodel.wl_utils import assert_partitions_equivalent

E1 = np.array([(1,0,0), (1,2,1), (3,1,2), (2,1,2)], dtype=np.int64)
E2 = np.array([(1,0,0), (2,1,1), (3,2,2), (2,1,3), (3,2,4)], dtype=np.int64)

nodes1 = [0, 1, 1, 1, 2, 2, 3]
times1 = [0, 0, 1, 2, 1, 2, 2]
nodes2 = [0, 1, 1, 1, 2, 2, 2, 2, 3, 3]
times2 = [0, 0, 1, 3, 1, 2, 3, 4, 2, 4]


class TestTemporalWL(unittest.TestCase):
    def test_compute_d_rounds_1(self):
        colors, nodes, times = compute_d_rounds(E1, 4, d=3, h=-1, seed=0)
        self.assertEqual(len(colors), 2)
        assert_partitions_equivalent(colors[0], np.zeros(7, dtype=int))
        assert_partitions_equivalent(colors[1], [1,2,2,2,1,0,0])
        assert_array_equal(nodes, nodes1)
        assert_array_equal(times, times1)

    def test_compute_d_rounds_1_1(self):
        colors, nodes, times = compute_d_rounds(E1, num_nodes=5, d=3, h=-1, seed=0)
        self.assertEqual(len(colors), 2)
        assert_partitions_equivalent(colors[0], np.zeros(7, dtype=int))
        assert_partitions_equivalent(colors[1], [1,2,2,2,1,0,0])
        assert_array_equal(nodes, nodes1)
        assert_array_equal(times, times1)


    def test_compute_d_rounds_2(self):
        colors, nodes, times = compute_d_rounds(E2, 4, d=10, h=-1, seed=0)
        self.assertEqual(len(colors), 3)
        assert_partitions_equivalent(colors[0], np.zeros(10, dtype=int))
        assert_partitions_equivalent(colors[1], [1, 2, 2, 1, 2, 2, 1, 1, 0, 0])
        assert_array_equal(colors[1][-2:], [0, 0]) # make sure degree zero nodes are assigned color 0
        assert_partitions_equivalent(colors[2], [1, 3, 3, 2, 5, 5, 4, 4, 0, 0])
        assert_array_equal(colors[2][-2:], [0, 0]) # make sure degree zero nodes are assigned color 0
        assert_array_equal(nodes, nodes2)
        assert_array_equal(times, times2)


    def test_compute_d_rounds_finite_h1(self):
        colors, nodes, times = compute_d_rounds(E2, 4, d=10, h=1, seed=0)
        self.assertEqual(len(colors), 4)
        assert_partitions_equivalent(colors[0], np.zeros(10, dtype=int))
        assert_partitions_equivalent(colors[1], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        assert_partitions_equivalent(colors[2], [2, 2, 2, 2, 1, 1, 1, 1, 0, 0])
        assert_partitions_equivalent(colors[3], [1, 3, 3, 3, 2, 2, 2, 2, 0, 0])
        assert_array_equal(nodes, nodes2)
        assert_array_equal(times, times2)


    def test_compute_d_rounds_finite_h0_1(self):
        """Test for h=0"""
        colors, nodes, times = compute_d_rounds(E1, 4, d=10, h=0, seed=0)
        self.assertEqual(len(colors), 2)
        assert_partitions_equivalent(colors[0], np.zeros(7, dtype=int))
        assert_partitions_equivalent(colors[1], [2, 1, 1, 3, 2, 1, 1])
        assert_array_equal(nodes, nodes1)
        assert_array_equal(times, times1)

    def test_compute_d_rounds_finite_h0_2(self):
        """Test for h=0"""
        colors, nodes, times = compute_d_rounds(E2, 4, d=10, h=0, seed=0)
        self.assertEqual(len(colors), 2)
        assert_partitions_equivalent(colors[0], np.zeros(10, dtype=int))
        assert_partitions_equivalent(colors[1], [2, 1, 2, 2, 1, 2, 1, 2, 1, 1])
        assert_array_equal(nodes, nodes2)
        assert_array_equal(times, times2)

    def test_compute_d_rounds_finite_h2_2(self):
        """Test for h=2"""
        colors, nodes, times = compute_d_rounds(E2, 4, d=10, h=2, seed=0)
        self.assertEqual(len(colors), 4)
        assert_partitions_equivalent(colors[0], np.zeros(10, dtype=int))
        assert_partitions_equivalent(colors[1], [1, 1, 2, 1, 1, 2, 1, 1, 0, 0])
        assert_partitions_equivalent(colors[2], [1, 1, 2, 1, 3, 4, 3, 3, 0, 0])
        assert_partitions_equivalent(colors[3], [3, 1, 2, 1, 4, 5, 4, 4, 0, 0])
        assert_array_equal(nodes, nodes2)
        assert_array_equal(times, times2)




class TestTemporalStruct(unittest.TestCase):
    def check_partitions_agree(self, s, times, solutions, check_global=True):
        global_partitions = []
        for t, sol in zip(times, solutions):
            s.advance_time(t)
            if check_global:
                global_partitions.append(s.current_colors.copy())
            assert_partitions_equivalent(s.current_colors, sol)
        if check_global:
            sols = np.array(solutions, dtype=np.int64).ravel()
            global_array = np.array(global_partitions, dtype=np.int64).ravel()
            assert_partitions_equivalent(sols, global_array)


    def test_global_fill_h1_d1(self):
        h=1
        s = TemporalColorsStruct(*_compute_d_rounds(E2, 4, d=-1, h=h))
        s.reset_colors(d=1, h = h)
        solutions = [
            [1,1,0,0],
            [0,1,1,0],
            [0,1,1,0],
            [0,1,1,0],
            [0,0,1,0]
        ]
        self.check_partitions_agree(s, range(5), solutions)


    def test_global_fill_h2_d1(self):
        h=2
        s = TemporalColorsStruct(*_compute_d_rounds(E2, 4, d=-1, h=h))
        s.reset_colors(d=1, h = h)
        solutions = [
            [1,1,1,0],
            [0,2,1,0],
            [0,1,2,0],
            [0,1,1,0],
            [0,0,1,0]
        ]
        self.check_partitions_agree(s, range(5), solutions)


    def test_global_fill_h2_d2(self):
        h=2
        s = TemporalColorsStruct(*_compute_d_rounds(E2, 4, d=-1, h=h))
        s.reset_colors(d=2, h = h)
        solutions = [
            [1,1,2,0],
            [0,3,2,0],
            [0,1,4,0],
            [0,1,2,0],
            [0,0,2,0]
        ]
        self.check_partitions_agree(s, range(5), solutions)
        s.reset_colors(d=2, h=h)
        self.check_partitions_agree(s, range(5), solutions)
        s.reset_colors(d=2, h=h)
        s.advance_time(0)
        # #assert_partitions_equivalent(s.current_colors, [1,1,1,0])
        # print("<<<<<", s.current_colors)

        # s.advance_time(1)
        # #assert_partitions_equivalent(s.current_colors, [0,2,1,0])
        # print("<<<<<", s.current_colors)

        # s.advance_time(2)
        # #assert_partitions_equivalent(s.current_colors, [0,1,2,0])
        # print("<<<<<", s.current_colors)

        # s.advance_time(3)
        # #assert_partitions_equivalent(s.current_colors, [0,1,1,0])
        # print("<<<<<", s.current_colors)

        # s.advance_time(4)
        # #assert_partitions_equivalent(s.current_colors, [0,0,1,0])
        # print("<<<<<", s.current_colors)


    def test_global_fill_h1_d1_local(self):
        h=1
        s = TemporalColorsStruct(*_compute_d_rounds(E2, 4, d=-1, h=h))
        s.reset_colors(d=1, h = h, mode="local")
        solutions = [
            [1,1,0,0],
            [0,1,1,0],
            [0,1,1,0],
            [0,1,1,0],
            [0,0,1,0]
        ]
        self.check_partitions_agree(s, range(5), solutions, check_global=False)


    def test_global_fill_h2_d2_local(self):
        h=2
        s = TemporalColorsStruct(*_compute_d_rounds(E2, 4, d=-1, h=h))
        s.reset_colors(d=2, h = h, mode="local")
        solutions = [
            [1,1,2,0],
            [0,3,2,0],
            [0,1,4,0],
            [0,1,2,0],
            [0,0,2,0]
        ]
        self.check_partitions_agree(s, range(5), solutions, check_global=False)
        s.reset_colors(d=2, h=h)
        self.check_partitions_agree(s, range(5), solutions, check_global=False)


if __name__ == '__main__':
    unittest.main()
