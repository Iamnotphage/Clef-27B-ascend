import copy
import unittest
from contract import MAGIC, checksum, intervals, parse_diag


class ContractTests(unittest.TestCase):
    def good(self, m=967, cores=48, ar=1, rank=0):
        rows = [[0] * 32 for _ in range(48)]
        for i, (a, b) in enumerate(intervals(m, cores)):
            rows[i][:13] = [MAGIC, 1, i, cores, a, b, ar,
                           rank if ar and i == 0 else 0xFFFFFFFF,
                           2 if ar and i == 0 else 0xFFFFFFFF, 0, 0, m, 5120]
            rows[i][31] = checksum(rows[i][:31])
        return rows

    def test_partition_exhaustive(self):
        # All supported M/core combinations, including zero-row cores and tails.
        for m in range(1, 16385):
            for cores in (1, 2, 3, 7, 24, 48):
                ranges = intervals(m, cores)
                self.assertEqual(ranges[0][0], 0)
                self.assertEqual(ranges[-1][1], m)
                self.assertEqual(sum(b-a for a, b in ranges), m)
                self.assertTrue(all(ranges[i][1] == ranges[i+1][0] for i in range(cores-1)))

    def test_valid(self):
        for m in (1, 7, 8, 307, 967, 1595, 16384):
            for cores in (1, 3, 24, 48):
                for ar in (0, 1):
                    for rank in (0, 1):
                        self.assertTrue(parse_diag(self.good(m, cores, ar, rank), m, cores, ar, rank)["pass"])

    def test_corrupt_every_word(self):
        for w in range(32):
            rows = self.good()
            rows[0][w] ^= 1
            self.assertFalse(parse_diag(rows, 967, 48, 1, 0)["pass"])

    def test_incomplete_and_extra_slot(self):
        rows = self.good(967, 1)
        rows[1][0] = 1
        self.assertFalse(parse_diag(rows, 967, 1, 1, 0)["pass"])
        rows = self.good()
        rows[0] = [0]*32
        self.assertFalse(parse_diag(rows, 967, 48, 1, 0)["pass"])


if __name__ == "__main__":
    unittest.main()
