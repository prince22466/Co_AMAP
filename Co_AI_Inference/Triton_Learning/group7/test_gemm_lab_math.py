#!/usr/bin/env python3
"""Group7 CPU-only scheduling and split-K coverage tests (no Triton needed)."""
import math
import unittest

from gemm_lab_common import (
    group_tile_coords, parse_tiles, persistent_tiles, split_k_intervals,
)


class SchedulingTests(unittest.TestCase):
    def test_all_grouped_grids_cover_unique_tiles(self):
        for nm in (1, 2, 3, 5, 11):
            for nn in (1, 2, 4, 7):
                expected = {(m, n) for m in range(nm) for n in range(nn)}
                for group in (1, 2, 4, 8, 16):
                    actual = [group_tile_coords(i, nm, nn, group)
                              for i in range(nm * nn)]
                    self.assertEqual(len(actual), len(set(actual)))
                    self.assertEqual(set(actual), expected)
                    if group == 1:
                        self.assertEqual(actual, [(m, n) for m in range(nm)
                                                  for n in range(nn)])

    def test_split_k_intervals_exact(self):
        for k in (1, 17, 31, 32, 33, 63, 96, 97, 1023, 1024, 1025):
            for bk in (16, 32, 64):
                for splits in (1, 2, 4, 8, 16):
                    intervals = split_k_intervals(k, bk, splits)
                    covered = []
                    for start, stop in intervals:
                        self.assertLessEqual(start, stop)
                        self.assertEqual(start % bk, 0 if start < k else k % bk)
                        covered.extend(range(start, stop))
                    self.assertEqual(covered, list(range(k)))

    def test_persistent_cover_exactly_once(self):
        for total in (1, 2, 3, 7, 64, 129):
            for workers in (1, 2, 8, 16, 64, 128):
                assigned = persistent_tiles(total, workers)
                self.assertEqual(len(assigned), min(total, workers))
                flat = [tile for worker in assigned for tile in worker]
                self.assertEqual(sorted(flat), list(range(total)))
                self.assertEqual(len(flat), len(set(flat)))

    def test_tile_parser(self):
        self.assertEqual(parse_tiles('32x64x32,64x64x32,32x64x32'),
                         [(32, 64, 32), (64, 64, 32)])
        for bad in ('', '32x64', '32x48x32', '0x64x32', '8x64x32'):
            with self.assertRaises(ValueError):
                parse_tiles(bad)

    def test_splitk_math(self):
        # Sequential CPU gemm vs sum over the same independent K partitions.
        m, n, k = 5, 7, 37
        a = [[((i * 7 + t) % 13 - 5) / 9 for t in range(k)] for i in range(m)]
        b = [[((t * 3 + j * 5) % 17 - 6) / 11 for j in range(n)] for t in range(k)]
        direct = [[sum(a[i][t] * b[t][j] for t in range(k)) for j in range(n)]
                  for i in range(m)]
        for splits in (1, 2, 4, 8):
            pieces = split_k_intervals(k, 16, splits)
            combined = [[sum(sum(a[i][t] * b[t][j] for t in range(lo, hi))
                             for lo, hi in pieces) for j in range(n)] for i in range(m)]
            for i in range(m):
                for j in range(n):
                    self.assertTrue(math.isclose(direct[i][j], combined[i][j],
                                                 abs_tol=1e-10, rel_tol=1e-10))


if __name__ == '__main__':
    unittest.main(verbosity=2)
