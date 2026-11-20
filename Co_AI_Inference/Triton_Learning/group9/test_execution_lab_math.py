#!/usr/bin/env python3
"""CPU-only regression tests for Group9. No GPU and no Triton needed."""
import unittest
import numpy as np

from execution_reference import (
    cpu_affine_sequence, cpu_fusion, cpu_launch, cpu_persistent,
    parse_positive_list, persistent_tile_assignment,
)


class ExecutionLabReferenceTests(unittest.TestCase):
    def test_launch_reference(self):
        x = np.array([-2.5, 0, 0.25, 0.7], dtype=np.float32)
        np.testing.assert_array_equal(cpu_launch(x), x + np.float32(0.25))

    def test_fusion_equivalence(self):
        x = np.linspace(-0.7, 0.7, 101, dtype=np.float32)
        temp = (x * np.float32(1.125) + np.float32(0.25)).astype(np.float32)
        separate = (temp * temp + x).astype(np.float32)
        np.testing.assert_array_equal(cpu_fusion(x), separate)

    def test_affine_chain(self):
        x = np.array([-0.5, 0, 0.5], dtype=np.float32)
        out = cpu_affine_sequence(x, 9)
        ref = x.copy()
        for _ in range(9):
            ref = np.float32(ref * np.float32(0.987) + np.float32(0.002))
        np.testing.assert_array_equal(out, ref)

    def test_persistent_math(self):
        x = np.linspace(-0.3, 0.7, 37, dtype=np.float32)
        v = cpu_persistent(x, 12)
        ref = x.astype(np.float64)
        for _ in range(12):
            ref = ref * 0.999 + 0.0001
        np.testing.assert_allclose(v, ref.astype(np.float32), atol=0, rtol=0)

    def test_persistent_assignment_odd_tail(self):
        tiles = (1003 + 255) // 256
        for worker_count in (1, 2, 4, 32):
            groups = persistent_tile_assignment(1003, 256, worker_count)
            covered = [t for g in groups for t in g]
            self.assertEqual(sorted(covered), list(range(tiles)))
            self.assertEqual(len(covered), len(set(covered)))

    def test_invalid_inputs(self):
        self.assertEqual(parse_positive_list('1,2,4,2', 'sizes'), [1, 2, 4])
        with self.assertRaises(ValueError):
            parse_positive_list('1,0', 'sizes')
        with self.assertRaises(ValueError):
            parse_positive_list('3', 'sizes', powers_of_two=True)
        with self.assertRaises(ValueError):
            cpu_affine_sequence([1], 0)
        with self.assertRaises(ValueError):
            persistent_tile_assignment(0, 256, 8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
