# pylint: disable=missing-function-docstring, missing-class-docstring
import unittest
import numpy as np
from numpy.testing import assert_array_equal
from tnestmodel.temp_fast_graph import TempFastGraph, SparseTempFastGraph
from tnestmodel.temp_fast_graph import get_rolling_max_degree



def to_edge_sets(e1, e2):
    return set(map(tuple, e1)), set(map(tuple, e2))


class TestTFastGraph(unittest.TestCase):
    def temp_fast_graph_test(self, l_edges, num_nodes, result_edges, is_directed):

        l_edges = [np.array(edges, dtype=np.uint32) for edges in l_edges]
        G_t = TempFastGraph(l_edges, is_directed=is_directed)
        result_edges = np.array(result_edges, dtype=np.uint32)

        G = G_t.get_causal_completion()
        self.assertEqual(G.num_nodes, num_nodes)
        assert_array_equal(result_edges, G.edges)

    def sparse_temp_fast_graph_test(self, l_edges, num_nodes, result_edges, is_directed, dense_edges=None, horizons=(-1,)):
        if isinstance(l_edges, np.ndarray) and l_edges.shape[1]==3:
            G_t = SparseTempFastGraph.from_temporal_edges(l_edges, is_directed=is_directed)
        else:
            l_edges = [np.array(edges, dtype=np.uint32) for edges in l_edges]
            G_t = SparseTempFastGraph(l_edges, is_directed=is_directed)
        result_edges = np.array(result_edges, dtype=np.uint32)
        with self.subTest(causal_completion="sparse"):
            G = G_t.get_sparse_causal_completion()
            self.assertEqual(G.num_nodes, num_nodes)
            # print("sparse", repr(G.edges))
            assert_array_equal(result_edges, G.edges)
            del G
        with self.subTest(causal_completion="dense"):
            self.assertEqual(len(dense_edges), len(horizons))
            for d_edges, h in zip(dense_edges, horizons):
                with self.subTest(horizon=h):
                    G_d = G_t.get_dense_causal_completion(h=h)
                    self.assertEqual(G_d.num_nodes, G_t.num_nodes*G_t.num_times)
                    self.assertTupleEqual(G_d.identifiers.shape, (G_t.num_nodes*G_t.num_times, 2))
                    #print(repr(G_d.edges))
                    self.assertSetEqual(*to_edge_sets(d_edges, G_d.edges))

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
                                  is_directed = True,
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
        with self.subTest(from_="graphs"):
            self.sparse_temp_fast_graph_test([[[0,1]],
                                    [[1,2]]],
                                    4,
                                    [[0,1], [1, 3], [2, 3]],
                                    is_directed = True,
                                    dense_edges=[[[0, 1],
                                                  [4, 5]],
                                                  [[0, 1],
                                                  [1, 5],
                                                  [4, 5]],
                                                  [[0, 1],
                                                  [1, 5],
                                                  [4, 5]],],
                                    horizons=[0, -1, 1]
                                    )
        with self.subTest(from_="edges"):
            self.sparse_temp_fast_graph_test(np.array([[0,1,0],
                                                [1,2,1]],dtype=int),
                                    4,
                                    [[0,1], [1, 3], [2, 3]],
                                    is_directed = True,
                                    dense_edges=[[[0, 1],
                                                  [4, 5]],
                                                  [[0, 1],
                                                  [1, 5],
                                                  [4, 5]],
                                                  [[0, 1],
                                                  [1, 5],
                                                  [4, 5]],],
                                    horizons=[0, -1, 1]
                                    )


    def test_sparse_big_graph_2(self):
        self.sparse_temp_fast_graph_test([[[0,1]],
                                   [[1,2]],
                                   [[1,2]]],
                                  5,
                                  [[0,1], [1, 4], [1,4], [2,4], [2,4], [3,4]],
                                  is_directed = True,
                                  dense_edges= [[[0, 1],
                                                 [4, 5],
                                                 [7, 8]],
                                                [[0, 1],
                                                 [1, 5],
                                                 [4, 5],
                                                 [4, 8],
                                                 [7, 8]],
                                                 [[0, 1],
                                                 [1, 5],
                                                 [4, 5],
                                                 [1, 8],
                                                 [4, 8],
                                                 [7, 8]],
                                                 [[0, 1],
                                                 [1, 5],
                                                 [4, 5],
                                                 [1, 8],
                                                 [4, 8],
                                                 [7, 8]],],
                                    horizons=[0,1,2,-1]
                                  )


    def test_sparse_big_graph_3(self):
        d_edges = [[[0, 1],
                        [1, 2],
                        [4, 5],
                        [6, 7],
                        [7, 8],],
                        [[0, 1],
                        [1, 2],
                        [1, 5],
                        [4, 5],
                        [3, 7],
                        [6, 7],
                        [4, 8],
                        [7, 8],],
                        [[0, 1],
                        [1, 2],
                        [1, 5],
                        [4, 5],
                        [0, 7],
                        [3, 7],
                        [6, 7],
                        [1, 8],
                        [4, 8],
                        [7, 8],],
                        [[0, 1],
                        [1, 2],
                        [1, 5],
                        [4, 5],
                        [0, 7],
                        [3, 7],
                        [6, 7],
                        [1, 8],
                        [4, 8],
                        [7, 8],],
                        ]
        with self.subTest(from_="graphs"):
            self.sparse_temp_fast_graph_test([[[0,1], [1,2]],
                                    [[1,2]],
                                    [[0,1], [1,2]]],
                                    6,
                                    [[0, 1], [0, 4], [1, 5], [1,5], [1,5], [2, 5], [2,5], [3,4], [4,5]],
                                    is_directed = True,
                                    dense_edges= d_edges,
                                    horizons=[0,1,2,-1]
                                    )
        with self.subTest(from_="edges"):
            self.sparse_temp_fast_graph_test(np.array([[0,1,0], [1,2,0],
                                    [1,2,1],
                                    [0,1,2], [1,2,2]],dtype=int),
                                    6,
                                    [[0, 1], [0, 4], [1, 5], [1,5], [1,5], [2, 5], [2,5], [3,4], [4,5]],
                                    is_directed = True,
                                    dense_edges= d_edges,
                                    horizons=[0,1,2,-1]
                                    )


    def test_sparse_big_graph_undir_1(self):
        dense_edges = [[[0, 1],
                        [1, 0],
                        [1, 2],
                        [2, 1],
                        [4, 5],
                        [5, 4],],
                        [[0, 1],
                        [1, 0],
                        [1, 2],
                        [2, 1],
                        [1, 5],
                        [2, 4],
                        [4, 5],
                        [5, 4],],
                        [[0, 1],
                        [1, 0],
                        [1, 2],
                        [2, 1],
                        [1, 5],
                        [2, 4],
                        [4, 5],
                        [5, 4],],]
        self.sparse_temp_fast_graph_test([[[0,1], [1,2]],
                                   [[1,2]]],
                                  5,
                                  [[0, 1], [1, 0], [1,2], [1,4], [2,1], [2,3], [3,4], [4,3]],
                                  is_directed = False,
                                  dense_edges=dense_edges,
                                  horizons=[0,1,-1]
                                  )


    def test_sparse_big_graph_undir_2(self):
        dense_edges = [[[0, 1],
                        [1, 0],
                        [1, 2],
                        [2, 1],
                        [4, 5],
                        [5, 4],
                        [6, 8],
                        [8, 6]],

                        [[0, 1],
                        [1, 0],
                        [1, 2],
                        [2, 1],
                        [4, 5],
                        [1, 5],
                        [5, 4],
                        [2, 4],
                        [6, 8],
                        [3, 8],
                        [8, 6],
                        [5, 6]],

                        [[0, 1],
                        [1, 0],
                        [1, 2],
                        [2, 1],
                        [4, 5],
                        [1, 5],
                        [5, 4],
                        [2, 4],
                        [6, 8],
                        [0, 8],
                        [3, 8],
                        [8, 6],
                        [2, 6],
                        [5, 6]],]
        self.sparse_temp_fast_graph_test([[[0,1], [1,2]],
                                   [[1,2]],
                                   [[0,2]]],
                                  7,
                                   [[0, 1],
                                    [0, 6],
                                    [1, 0],
                                    [1, 2],
                                    [1, 4],
                                    [2, 1],
                                    [2, 3],
                                    [2, 5],
                                    [3, 4],
                                    [4, 3],
                                    [4, 5],
                                    [5, 6],
                                    [6, 5]],
                                  is_directed = False,
                                  dense_edges= dense_edges,
                                    horizons=[0,1,-1]
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





class TestTFastGraphHelpers(unittest.TestCase):
    def test_rolling_max_degree(self):
        tmp_degrees = [[0,1,2], [2,2,0], [0,0,1], [1,1,1], [1,0,1]]
        l_degrees = [tmp_degrees, tmp_degrees]
        l_mapping = [[0,1,2]]*5
        results = {
            0 : [2,2,2],
            1 : [2,3,2],
            2 : [2,3,3],
            3 : [3,4,4],
            4 : [4,4,5]
        }
        for h, expected_result in results.items():
            res1, res2 = get_rolling_max_degree(l_degrees, l_mapping, False, h=h, num_nodes=3)
            assert_array_equal(res1, res2)
            assert_array_equal(res1, expected_result, err_msg=f"h={h}")


    def test_rolling_max_degree_2(self):
        tmp_degrees = [[1,2], [2,2], [1], [1,1,1], [1,1]]
        l_degrees = [tmp_degrees, tmp_degrees]
        l_mapping = [[1,2], [0,1], [2], [0,1,2], [0,2]]
        results = {
            0 : [2,2,2],
            1 : [2,3,2],
            2 : [2,3,3],
            3 : [3,4,4],
            4 : [4,4,5],
            5 : [4,4,5],
        }
        for h, expected_result in results.items():
            res1, res2 = get_rolling_max_degree(l_degrees, l_mapping, False, h=h, num_nodes=3)
            assert_array_equal(res1, res2)
            assert_array_equal(res1, expected_result, err_msg=f"h={h}")


    def test_rolling_max_degree_dir(self):
        tmp_degrees1 = [[1,2], [2,2], [1], [1,1,1], [1,1]]
        tmp_degrees2 = [[0,4], [1,0], [1], [1,2,3], [0,3]]
        l_degrees = [tmp_degrees1, tmp_degrees2]
        l_mapping = [[1,2], [0,1], [2], [0,1,2], [0,2]]
        results = {
            0 : ([2,2,2], [1,2,4]),
            1 : ([2,3,2], [1,2,4]),
            2 : ([2,3,3], [1,2,5]),
            3 : ([3,4,4], [2,2,8]),
            4 : ([4,4,5], [2,2,11]),
            5 : ([4,4,5], [2,2,11]),
        }
        for h, (expected_result1, expected_result2) in results.items():
            res1, res2 = get_rolling_max_degree(l_degrees, l_mapping, True, h=h, num_nodes=3)
            assert_array_equal(res1, expected_result1, err_msg=f"h={h}")
            assert_array_equal(res2, expected_result2, err_msg=f"h={h}")


    def test_rolling_max_degree_3(self):
        tmp_degrees = [[0,1,2], [0,0,0], [0,0,0], [2,2,0], [0,0,1], [0,0,0], [0,0,0], [1,1,1], [1,0,1]]
        l_degrees = [tmp_degrees, tmp_degrees]
        l_mapping = [[0,1,2]]*9
        results = {
            0 : [2,2,2],
            1 : [2,2,2],
            2 : [2,2,2],
            3 : [2,3,2],
            4 : [2,3,3],
            5 : [2,3,3],
            6 : [2,3,3],
            7 : [3,4,4],
            8 : [4,4,5],
            9 : [4,4,5],
        }
        for h, expected_result in results.items():
            res1, res2 = get_rolling_max_degree(l_degrees, l_mapping, False, h=h, num_nodes=3)
            assert_array_equal(res1, res2)
            assert_array_equal(res1, expected_result, err_msg=f"h={h}")





class TestTFastGraphUtility(unittest.TestCase):
    def test_compute_for_each_slice(self):
        E = np.array([[0,1,0],[0,2,0], [1,2,1]], dtype=np.int64)
        G = SparseTempFastGraph.from_temporal_edges(E, is_directed=False)
        def num_edges(G, _):
            return len(G.edges)
        times, num_edges = G.compute_for_each_slice(num_edges, dtype=np.int64)
        assert_array_equal(times, [0,1])
        assert_array_equal(num_edges, [2,1])

    def test_compute_for_each_slice2(self):
        E = np.array([[0,1,3],[0,2,3], [1,2,12]], dtype=np.int64)
        G = SparseTempFastGraph.from_temporal_edges(E, is_directed=False)
        def num_edges(G, t):
            return len(G.edges)*t
        times, num_edges = G.compute_for_each_slice(num_edges, dtype=np.int64)
        assert_array_equal(times, [3,12])
        assert_array_equal(num_edges, [6,12])


if __name__ == '__main__':
    unittest.main()
