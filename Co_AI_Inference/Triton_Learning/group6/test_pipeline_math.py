#!/usr/bin/env python3
"""Pure CPU sanity tests: run without CUDA, Triton, or a GPU."""
import unittest

import numpy as np

from pipeline_lab_common import cpu_fma_series, positive_ints, prefetch_schedule


class PipelineReferenceTests(unittest.TestCase):
    def test_serial_vs_early_load_order(self):
        rng = np.random.default_rng(7)
        a = rng.normal(size=(17, 11)).astype(np.float32) * 0.1
        b = rng.normal(size=(17, 11)).astype(np.float32) * 0.1
        # Load order cannot affect independent stream mathematical results.
        serial = cpu_fma_series(a, 5) + cpu_fma_series(b, 5)
        early = cpu_fma_series(b, 5) + cpu_fma_series(a, 5)
        np.testing.assert_array_equal(serial, early)
        self.assertTrue(np.isfinite(serial).all())

    def test_prefetch_each_index_exactly_once(self):
        for steps in (4, 8, 17):
            for distance in (0, 1, 2, 4):
                with self.subTest(steps=steps, distance=distance):
                    fetched, consumed = prefetch_schedule(steps, distance)
                    self.assertEqual(fetched, list(range(steps)))
                    self.assertEqual(consumed, list(range(steps)))
                    x = np.arange(steps * 3, dtype=np.float64).reshape(steps, 3) / 20
                    np.testing.assert_array_equal(
                        cpu_fma_series(x[consumed], 3),
                        cpu_fma_series(x, 3))

    def test_numeric_reference_hand_calculation(self):
        x = np.array([[0.5, -0.5], [0.25, -0.25]], dtype=np.float32)
        got = cpu_fma_series(x, 1)
        want = np.array([0.5 * 0.2 * 0.75 + 0.25 * 0.2,
                         -0.5 * 0.2 * 0.75 - 0.25 * 0.2])
        np.testing.assert_allclose(got, want)

    def test_reject_bad_schedules_and_options(self):
        for arg in ((3, 4), (0, 1), (10, 3)):
            with self.assertRaises(ValueError):
                prefetch_schedule(*arg)
        self.assertEqual(positive_ints("1,2,2,4", name="stages"), [1, 2, 4])
        with self.assertRaises(ValueError):
            positive_ints("1,-2", name="stages")


if __name__ == "__main__":
    unittest.main(verbosity=2)
