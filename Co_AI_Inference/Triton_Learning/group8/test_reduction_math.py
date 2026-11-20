#!/usr/bin/env python3
"""CPU-only mathematical and sweep-validation tests; no Triton required."""
import math
import random
import unittest

from reduction_lab_common import cpu_reference, parse_list
from reduction_reference import (
    chunked_sum, online_softmax, online_stats, sequential_sum, stable_softmax,
)


class ReductionReferenceTests(unittest.TestCase):
    def test_sequential_and_chunked_sum(self):
        random.seed(34)
        for n in (1, 17, 64, 137, 1024):
            row = [random.uniform(-3, 3) for _ in range(n)]
            expected = math.fsum(row)
            self.assertAlmostEqual(sequential_sum(row), expected, places=10)
            for chunk in (1, 2, 8, 32, 64, 128):
                self.assertAlmostEqual(chunked_sum(row, chunk), expected, places=10)

    def test_online_softmax_odd_tails_and_extremes(self):
        random.seed(9)
        for n in (1, 3, 17, 137, 257, 1024):
            row = [random.uniform(-7, 7) for _ in range(n)]
            row[0] = 1000.0
            if n > 1:
                row[-1] = -1000.0
            baseline = stable_softmax(row)
            for chunk in (1, 2, 32, 64, 128, 256):
                got = online_softmax(row, chunk)
                self.assertEqual(len(got), n)
                self.assertTrue(math.isclose(sum(got), 1, rel_tol=0, abs_tol=1e-12))
                for p, q in zip(got, baseline):
                    self.assertAlmostEqual(p, q, places=12)
                m, s = online_stats(row, chunk)
                self.assertTrue(math.isfinite(m))
                self.assertTrue(s > 0)

    def test_softmax_translation_invariance(self):
        values = [1.0, -3.0, 1.5, 0.0, 4.0]
        for shift in (-80, 0, 80):
            one = stable_softmax(values)
            two = stable_softmax([v + shift for v in values])
            for a, b in zip(one, two):
                self.assertAlmostEqual(a, b, places=12)

    def test_parse_warps_and_chunk_sizes(self):
        self.assertEqual(parse_list('4,8,4', 'warps'), [4, 8])
        self.assertEqual(parse_list('32,64,128', 'chunks'), [32, 64, 128])
        for broken in ('0', '-4', '3', 'four', ''):
            with self.assertRaises(ValueError):
                parse_list(broken, 'warps')

    def test_cpu_fp64_reference_on_explicit_values(self):
        # No Triton or GPU needed.  Exercises the exact reference-generation
        # helper used by all four labs, including nonzero cancellation cases.
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch not installed')
        x = torch.tensor([[1.0, -1.0, 3.0], [5.0, 6.0, -7.0]],
                         dtype=torch.float32)
        for lab in ('R01', 'R02'):
            expected = cpu_reference(torch, x, lab)
            self.assertEqual(expected.device.type, 'cpu')
            self.assertEqual(expected.dtype, torch.float32)
            torch.testing.assert_close(expected, torch.tensor([3.0, 4.0]))
        for lab in ('R03', 'R04'):
            expected = cpu_reference(torch, x, lab)
            self.assertEqual(expected.dtype, torch.float32)
            torch.testing.assert_close(expected.sum(dim=1), torch.ones(2),
                                       atol=2e-7, rtol=2e-7)
            reference = torch.softmax(x.double(), dim=-1).float()
            torch.testing.assert_close(expected, reference)
        with self.assertRaises(ValueError):
            cpu_reference(torch, x, 'R99')

    def test_cpu_fp64_reference_matches_reported_biv150_r01(self):
        # Regression for the reported rows=17, cols=137, seed=2026 failure.
        # Values below are actual Triton outputs from the BI-V150 log.
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch not installed')
        generator = torch.Generator(device='cpu').manual_seed(2026)
        x = torch.randn((17, 137), generator=generator,
                        dtype=torch.float32) * 1.25
        x[::11, 0] = 12.0
        x[::13, 1] = -12.0
        expected = cpu_reference(torch, x, 'R01')
        observed_first = torch.tensor([
            -8.864295959472656, -6.8500213623046875,
             9.088007926940918, -4.9479498863220215,
             1.3040738105773926, 11.741593360900879,
            -1.3324027061462402, -13.655158042907715,
        ])
        torch.testing.assert_close(expected[:8], observed_first,
                                   atol=5e-4, rtol=1e-4)
        self.assertAlmostEqual(expected.abs().max().item(),
                               31.35272979736328, delta=0.0005)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            stable_softmax([])
        with self.assertRaises(ValueError):
            online_softmax([], 2)
        with self.assertRaises(ValueError):
            chunked_sum([1.0], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
