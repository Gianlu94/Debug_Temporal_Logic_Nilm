import unittest

import numpy as np
import numpy.testing as npt

import nilm.data as data

class TestUKdaleFunctions(unittest.TestCase):
    
    def test_transform_ukdale_to_chain2(self):

        input_arr = np.array([[10,310,590,1100,45,68,112,225],
                              np.arange(8)]).T

        expected_arr = np.array([[10,310,310,1100,45,45,45,45],
                                  np.arange(8)]).T

        tested_arr = data.transform_ukdale_to_chain2(input_arr)
        npt.assert_array_equal(tested_arr, expected_arr)


class TestSlidingWindowTimeSeries(unittest.TestCase):
    
    def test_len(self):

        input_data = (np.arange(20), np.arange(20))
        ts = data.SlidingWindowTimeSeries(input_data[0], input_data[1], 3, 5, 1)

        npt.assert_array_equal(len(ts), 5 + 1)

        ts = data.SlidingWindowTimeSeries(input_data[0], input_data[1], 3, 5, 2)

        npt.assert_array_equal(len(ts), (5 + 1)/ 2)

        ts = data.SlidingWindowTimeSeries(input_data[0], input_data[1], 3, 5, 4)

        npt.assert_array_equal(len(ts), 2)

        ts = data.SlidingWindowTimeSeries(input_data[0], input_data[1], 3, 3, 1)

        npt.assert_array_equal(len(ts), 18)
        
        ts = data.SlidingWindowTimeSeries(input_data[0], input_data[1], 5, 5, 1)

        npt.assert_array_equal(len(ts), 16)

        ts = data.SlidingWindowTimeSeries(input_data[0], input_data[1], 9, 9, 1)

        npt.assert_array_equal(len(ts), 12)

    def test_getitem(self):

        input_data = (np.arange(20), np.arange(20))
        ts = data.SlidingWindowTimeSeries(input_data[0], input_data[1], 3, 5, 1)

        expected_data = (np.arange(5).reshape((1,-1)), np.arange(1, 4).reshape((1,-1)))
        output_data = ts[0]
        npt.assert_array_equal(output_data[0], expected_data[0])
        npt.assert_array_equal(output_data[1], expected_data[1])

        expected_data = (np.arange(3, 8).reshape((1,-1)), np.arange(4, 7).reshape((1,-1)))
        output_data = ts[1]
        npt.assert_array_equal(output_data[0], expected_data[0])
        npt.assert_array_equal(output_data[1], expected_data[1])

        expected_data = (np.arange(15, 20).reshape((1,-1)), np.arange(16, 19).reshape((1,-1)))
        output_data = ts[-1]
        npt.assert_array_equal(output_data[0], expected_data[0])
        npt.assert_array_equal(output_data[1], expected_data[1])
        
        input_data = (np.arange(20), np.arange(20))
        ts = data.SlidingWindowTimeSeries(input_data[0], input_data[1], 3, 5, 2)

        expected_data = (np.array([np.arange(5), np.arange(3, 8)]),
                         np.array([np.arange(1, 4), np.arange(4, 7)]))
        output_data = ts[0]
        npt.assert_array_equal(output_data[0], expected_data[0])
        npt.assert_array_equal(output_data[1], expected_data[1])


    def test_getitem2(self):
        input_data = (np.arange(20), np.arange(20))
        ts_small = data.SlidingWindowTimeSeries(input_data[0], input_data[1], 3, 5, 1)
        ts_big = data.SlidingWindowTimeSeries(input_data[0], input_data[1], 3, 10, 1)

        o_small = np.concatenate([ts_small[i][0][0,1:-1] for i in range(len(ts_small))])
        o_big = np.concatenate([ts_big[i][0][0,1:-1] for i in range(len(ts_big))])
        npt.assert_array_equal(o_small[:-2], o_big)

        l_small = np.concatenate([ts_small[i][1][0] for i in range(len(ts_small))])
        l_big = np.concatenate([ts_big[i][1][0] for i in range(len(ts_big))])
        npt.assert_array_equal(l_small[:-2], l_big)

        ts_small = data.SlidingWindowTimeSeries(input_data[0], input_data[1], 3, 5, 2)
        ts_big = data.SlidingWindowTimeSeries(input_data[0], input_data[1], 3, 10, 2)

        o_small = np.concatenate([ts_small[i][0][:,1:-1].flatten() for i in range(len(ts_small))])
        o_big = np.concatenate([ts_big[i][0][:,1:-1].flatten() for i in range(len(ts_big))])
        npt.assert_array_equal(o_small[:-2], o_big)

        t_small = np.concatenate([ts_small[i][1].flatten() for i in range(len(ts_small))])
        t_big = np.concatenate([ts_big[i][1].flatten() for i in range(len(ts_big))])
        npt.assert_array_equal(t_small[:-2], t_big)

    def test_getitem3(self):
        input_data = (np.arange(20), np.arange(20))
        ts  = data.SlidingWindowTimeSeries(input_data[0], input_data[1], 5, 5, 1)

        npt.assert_array_equal(ts[0][0], np.array([[0, 1, 2, 3, 4]]))
        npt.assert_array_equal(ts[0][1], np.array([[2]]))

        npt.assert_array_equal(ts[1][0], np.array([[1, 2, 3, 4, 5]]))
        npt.assert_array_equal(ts[1][1], np.array([[3]]))
        
        ts  = data.SlidingWindowTimeSeries(input_data[0], input_data[1], 5, 9, 1)

        npt.assert_array_equal(ts[0][0], np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]]))
        npt.assert_array_equal(ts[0][1], np.array([[2, 3, 4, 5, 6]]))

        npt.assert_array_equal(ts[1][0], np.array([[5, 6, 7, 8, 9, 10, 11, 12, 13]]))
        npt.assert_array_equal(ts[1][1], np.array([[7, 8, 9, 10, 11]]))


    def test_randomness(self):

        input_data = (np.arange(40), np.arange(40))
        ts1 = data.SlidingWindowTimeSeries(input_data[0], input_data[1], 3, 5, 2,
                                           shuffle=True, seed=1)
        ts1_clone = data.SlidingWindowTimeSeries(input_data[0], input_data[1], 3, 5, 2,
                                                 shuffle=True, seed=1)

        for i in range(len(ts1)):
            npt.assert_array_equal(ts1[i][0], ts1_clone[i][0])
            npt.assert_array_equal(ts1[i][1], ts1_clone[i][1])

        ts2 = data.SlidingWindowTimeSeries(input_data[0], input_data[1], 3, 5, 2,
                                           shuffle=True, seed=2)
        self.assertFalse(np.array_equal(ts1[0][0], ts2[0][0]))

        ts1.on_epoch_end()
        for i in range(len(ts1)):
            self.assertFalse(np.array_equal(ts1[i][0], ts1_clone[i][0]))

        ts1_clone.on_epoch_end()
        for i in range(len(ts1)):
            npt.assert_array_equal(ts1[i][0], ts1_clone[i][0])
            npt.assert_array_equal(ts1[i][1], ts1_clone[i][1])


class TestConcatenatedSequences(unittest.TestCase):

    def test_getitem(self):
        input_data = (np.arange(20), np.arange(20) + 20)
        
        cs = data.ConcatenatedSequences(input_data)

        for i in range(40):
            self.assertEqual(cs[i], i)

    def test_len(self):
        input_data = (np.arange(20),)
        
        cs = data.ConcatenatedSequences(input_data)

        self.assertEqual(len(cs), 20)

        input_data = (np.arange(20), np.arange(20) + 20)
        
        cs = data.ConcatenatedSequences(input_data)

        self.assertEqual(len(cs), 40)









