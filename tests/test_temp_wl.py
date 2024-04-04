# pylint: disable=missing-function-docstring, missing-class-docstring
import unittest

from tnestmodel.causal_completion import compare_wl_dense_sparse, compare_wl_dense_cumsum

from tnestmodel.temp_generation import temporal_Gnp




class TestDifferentWL(unittest.TestCase):
    def test_check_sparse_and_dense_produce_identical_wl_on_random_graphs(self,):
        """Tests agreement of dense and sparse causal completions on a lot of random graphs"""
        for n in range(3,21):
            with self.subTest(n=n):
                for seed in [0,100,1337,1231823129]:
                    with self.subTest(seed=seed):
                        for times in [5,10,20,100]:
                            with self.subTest(times=times):
                                G_temp = temporal_Gnp(n, 0.15, times, seed=seed)
                                if len(G_temp.slices)==0:
                                    continue
                                for h in [0,1,2,3,4,5,10,20,30,100]:
                                    with self.subTest(h=h):
                                        compare_wl_dense_cumsum(G_temp, h)
                                        compare_wl_dense_sparse(G_temp, h)

if __name__ == '__main__':
    unittest.main(failfast=True)