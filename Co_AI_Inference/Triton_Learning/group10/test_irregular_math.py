#!/usr/bin/env python3
"""CPU-only Group10 regression tests (no GPU/Triton required)."""
import unittest
import numpy as np
from irregular_reference import (
    gather_inputs, scatter_inputs, tail_reference, ragged_inputs,
    parse_int_csv, parse_floats,
)


class IrregularLabMathTests(unittest.TestCase):
    def test_gather_original_order_and_sorted_reconstruction(self):
        for pattern in ('uniform', 'hotset'):
            d = gather_inputs(1009, 8192, seed=2026, pattern=pattern)
            out = np.full(1009, np.nan, dtype=np.float32)
            out[d['original_positions']] = d['source'][d['sorted_indices']]
            np.testing.assert_array_equal(out, d['reference'])
            self.assertTrue(np.all(d['sorted_indices'][1:] >= d['sorted_indices'][:-1]))

    def test_scatter_bijection_and_sorted_reconstruction(self):
        for pattern in ('random', 'block_local'):
            d = scatter_inputs(1027, seed=2026, pattern=pattern)
            self.assertEqual(len(np.unique(d['destinations'])), 1027)
            direct = np.full(1027, np.nan, dtype=np.float32)
            direct[d['destinations']] = d['values']
            sorted_out = np.full(1027, np.nan, dtype=np.float32)
            sorted_out[d['sorted_destinations']] = d['values'][d['source_positions']]
            np.testing.assert_array_equal(direct, d['reference'])
            np.testing.assert_array_equal(sorted_out, d['reference'])
            self.assertTrue(np.array_equal(d['sorted_destinations'], np.arange(1027)))

    def test_tail_full_tiles_and_masked_reference(self):
        for n in (1, 127, 128, 129, 257, 4093):
            d = tail_reference(n, seed=2026)
            for block in (64, 128, 256):
                end = (n // block) * block
                simulated = np.full(n, np.nan, dtype=np.float32)
                x = d['x'].astype(np.float64)
                simulated[:end] = (x[:end] * 1.125 + 0.375).astype(np.float32)
                simulated[end:] = (x[end:] * 1.125 + 0.375).astype(np.float32)
                np.testing.assert_array_equal(simulated, d['reference'])

    def test_ragged_packed_and_padded_equivalence(self):
        for rate in (0.25, 0.75, 1.0):
            d = ragged_inputs(17, 137, rate, seed=2026)
            self.assertEqual(d['lengths'][0], 0)
            self.assertEqual(d['lengths'][1], 137)
            self.assertEqual(d['lengths'][2], 1)
            self.assertEqual(d['offsets'][-1], d['packed'].size)
            for row in range(17):
                a, b = int(d['offsets'][row]), int(d['offsets'][row+1])
                np.testing.assert_array_equal(d['packed'][a:b], d['padded'][row, :b-a])
                self.assertTrue(np.all(d['padded'][row, b-a:] == 0))
                ref = d['padded'][row, :b-a].astype(np.float64).sum()
                self.assertAlmostEqual(float(d['reference'][row]), float(np.float32(ref)), places=5)

    def test_ragged_small_rows_and_zeros(self):
        d = ragged_inputs(1, 1, 0.25, seed=2026)
        self.assertEqual(d['reference'].shape, (1,))
        self.assertEqual(d['padded'].shape, (1, 1))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            gather_inputs(0, 10, 1, 'uniform')
        with self.assertRaises(ValueError):
            gather_inputs(1, 10, 1, 'invalid')
        with self.assertRaises(ValueError):
            scatter_inputs(0, 1, 'random')
        with self.assertRaises(ValueError):
            ragged_inputs(17, 137, 0.0, 1)
        with self.assertRaises(ValueError):
            tail_reference(-1, 1)

    def test_argument_parsing(self):
        self.assertEqual(parse_int_csv('64, 128,64'), [64, 128])
        self.assertEqual(parse_floats('0.25, 0.75,0.25'), [0.25, 0.75])
        for value in ('', '1,,2', '4,', 'x'):
            with self.assertRaises(ValueError):
                parse_int_csv(value)
        for value in ('0.0', '1.1', '0.5,'):
            with self.assertRaises(ValueError):
                parse_floats(value)

    def test_seed_reproducibility(self):
        a = gather_inputs(121, 512, 7, 'uniform')
        b = gather_inputs(121, 512, 7, 'uniform')
        np.testing.assert_array_equal(a['indices'], b['indices'])
        a = scatter_inputs(121, 7, 'block_local')
        b = scatter_inputs(121, 7, 'block_local')
        np.testing.assert_array_equal(a['destinations'], b['destinations'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
