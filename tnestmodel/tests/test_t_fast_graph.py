# pylint: disable=missing-function-docstring, missing-class-docstring
import unittest
import numpy as np
from numpy.testing import assert_array_equal
from tnestmodel.t_fast_graph import TempFastGraph, SparseTempFastGraph







class TestTFastGraph(unittest.TestCase):
    def temp_fast_graph_test(self, l_edges, num_nodes, result_edges, is_directed):
        l_edges = [np.array(edges, dtype=np.uint32) for edges in l_edges]
        result_edges = np.array(result_edges, dtype=np.uint32)
        G_t = TempFastGraph(l_edges, is_directed=is_directed)
        G = G_t.get_causal_completion()
        self.assertEqual(G.num_nodes, num_nodes)
        assert_array_equal(result_edges, G.edges)

    def sparse_temp_fast_graph_test(self, l_edges, num_nodes, result_edges, is_directed):
        l_edges = [np.array(edges, dtype=np.uint32) for edges in l_edges]
        result_edges = np.array(result_edges, dtype=np.uint32)
        G_t = SparseTempFastGraph(l_edges, is_directed=is_directed)
        G = G_t.get_sparse_causal_completion()
        self.assertEqual(G.num_nodes, num_nodes)
        assert_array_equal(result_edges, G.edges)

    def test_edges(self):
        edges = np.array([[0,1]], dtype=np.uint32)
        G = TempFastGraph([edges.copy(), edges.copy()], is_directed=True)
        assert_array_equal(edges, G.slices[0].edges)
        assert_array_equal(edges, G.slices[1].edges)

    def test_num_nodes(self):
        edges1 = np.array([[0,1]], dtype=np.uint32)
        edges2 = np.array([[2,3]], dtype=np.uint32)
        G = TempFastGraph([edges1.copy(), edges2.copy()], is_directed=True)
        assert_array_equal(edges1, G.slices[0].edges)
        assert_array_equal(edges2, G.slices[1].edges)
        self.assertEqual(G.num_nodes, 4)

    def test_num_nodes2(self):
        edges0 = np.array([[0,1]], dtype=np.uint32)
        edges1 = np.array([[1,2]], dtype=np.uint32)
        G = TempFastGraph([edges0.copy(), edges1.copy()], is_directed=True)
        assert_array_equal(edges0, G.slices[0].edges)
        assert_array_equal(edges1, G.slices[1].edges)
        self.assertEqual(G.num_nodes, 3)

    def test_no_empty_time(self):
        edges0 = np.array([[0,1]], dtype=np.uint32)
        edges1 = np.empty((0,2), dtype=np.uint32)
        edges2 = np.array([[1,2]], dtype=np.uint32)
        with self.assertRaises(AssertionError):
            TempFastGraph([edges0.copy(), edges1.copy(), edges2.copy()], is_directed=True)

    def test_big_graph_1(self):
        self.temp_fast_graph_test([[[0,1]],
                                   [[1,2]]],
                                  6,
                                  [[0,1], [1, 5], [4,5]],
                                  is_directed = True
                                  )


    def test_big_graph_2(self):
        self.temp_fast_graph_test([[[0,1]],
                                   [[1,2]],
                                   [[1,2]]],
                                  9,
                                  [[0,1], [1, 5], [1,8], [4, 5], [4,8], [7,8]],
                                  is_directed = True
                                  )


    def test_big_graph_3(self):
        self.temp_fast_graph_test([[[0,1], [1,2]],
                                   [[1,2]],
                                   [[0,1], [1,2]]],
                                  9,
                                  [[0, 1], [1, 2], [1, 5], [0,7], [1,8], [4, 5],[3,7], [4,8], [6,7], [7,8]],
                                  is_directed = True
                                  )


    def test_big_graph_undir_1(self):
        self.temp_fast_graph_test([[[0,1], [1,2]],
                                   [[1,2]]],
                                  6,
                                  [[0, 1], [1, 2], [1,0], [2,1], [1,5], [2,4], [4,5], [5,4]],
                                  is_directed = False
                                  )



    ###### SPARSE BELOW #####
    def test_sparse_big_graph_1(self):
        self.sparse_temp_fast_graph_test([[[0,1]],
                                   [[1,2]]],
                                  4,
                                  [[0,1], [1, 3], [2, 3]],
                                  is_directed = True
                                  )


    def test_sparse_big_graph_2(self):
        self.sparse_temp_fast_graph_test([[[0,1]],
                                   [[1,2]],
                                   [[1,2]]],
                                  5,
                                  [[0,1], [1, 4], [1,4], [2,4], [2,4], [3,4]],
                                  is_directed = True
                                  )


    def test_sparse_big_graph_3(self):
        self.sparse_temp_fast_graph_test([[[0,1], [1,2]],
                                   [[1,2]],
                                   [[0,1], [1,2]]],
                                  6,
                                  [[0, 1], [0, 4], [1, 5], [1,5], [1,5], [2, 5], [2,5], [3,4], [4,5]],
                                  is_directed = True
                                  )


    def test_sparse_big_graph_undir_1(self):
        self.sparse_temp_fast_graph_test([[[0,1], [1,2]],
                                   [[1,2]]],
                                  5,
                                  [[0, 1], [1, 0], [1,2], [1,4], [2,1], [2,3], [3,4], [4,3]],
                                  is_directed = False
                                  )

    def test_causal_adjacency_the_same(self):
        edges0 = np.array([[2,1]], dtype=np.uint32)
        edges1 = np.array([[1,2]], dtype=np.uint32)
        edges2 = np.array([[2,3]], dtype=np.uint32)
        edges3 = np.array([[0,1],[1,2], [2,0]], dtype=np.uint32)
        G = TempFastGraph([edges0, edges1, edges2, edges3], is_directed=True)
        self.assertEqual((G.sparse_causal_adjacency()!=G.get_causal_completion().to_coo()).nnz, 0)


    def test_sparse_causal_adjacency_multi_edges(self):
        edges0 = np.array([[0,1], [1,2]], dtype=np.uint32)
        edges1 = np.array([[1,2]], dtype=np.uint32)
        edges2 = np.array([[0,1], [1,2]], dtype=np.uint32)
        G = SparseTempFastGraph([edges0, edges1, edges2], is_directed=True)
        coo = G.get_sparse_causal_completion().to_coo()
        assert_array_equal(coo.col, [1, 4, 5, 5, 5, 5, 5, 4, 5])
        assert_array_equal(coo.row, [0, 0, 1, 1, 1, 2, 2, 3, 4])
        assert_array_equal(coo.data, [1., 1., 1., 1., 1., 1., 1., 1., 1.])

        coo_dense = coo.todense()
        self.assertEqual(coo_dense[1,5], 3)
        self.assertEqual(coo_dense[2,5], 2)

if __name__ == '__main__':
    unittest.main()
