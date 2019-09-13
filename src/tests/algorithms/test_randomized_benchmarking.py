from __future__ import print_function
import unittest

from qtt.algorithms.randomized_benchmarking import CliffordRandomizedBenchmarkingSingleQubit


class TestCliffordRandomizedBenchMarkingSingleQubit(unittest.TestCase):

    def setUp(self):
        self.rb = CliffordRandomizedBenchmarkingSingleQubit()
        self.lengths = [2, 4, 8]
        self.num_seq = 2

    def test_generate_circuits(self):
        circuits = self.rb.generate_circuits(lengths=self.lengths, num_seq=self.num_seq)
        print(circuits)

    def test_generate_measurement_sequences(self):
        meas_seq = self.rb.generate_measurement_sequences(lengths=self.lengths, num_seq=self.num_seq)
        print(meas_seq)


if __name__ == '__main__':
    unittest.main()
