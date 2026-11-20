#!/usr/bin/env python3
"""CPU-only static/reference checks for Group5; no Triton or GPU required."""
import unittest
from smem_lab_common import parse_int_list, theoretical_gemm_operand_bytes


def source_offset_from_output(output_idx, m, n):
    row_out, col_out = divmod(output_idx, m)
    return col_out * n + row_out


class TestSharedMemoryLabMath(unittest.TestCase):
    def test_transpose_index(self):
        for m, n in [(3, 5), (33, 65), (64, 64)]:
            inp = list(range(m * n))
            a = [inp[source_offset_from_output(j, m, n)] for j in range(m * n)]
            b = [inp[i * n + j] for j in range(n) for i in range(m)]
            self.assertEqual(a, b)

    def test_modeled_tile_reuse(self):
        m = n = k = 512
        small = theoretical_gemm_operand_bytes(m, n, k, 32, 32)
        large = theoretical_gemm_operand_bytes(m, n, k, 64, 64)
        self.assertEqual(small, large * 2)
        self.assertGreater(small, 0)
        # Explicitly model padded tiles (not measured memory bytes).
        odd = theoretical_gemm_operand_bytes(65, 129, 63, 32, 32)
        self.assertEqual(odd, 3 * 5 * (32 * 63 + 32 * 63) * 2)

    def test_lists(self):
        self.assertEqual(parse_int_list("32, 64,32", power2=True), [32,64])
        for bad in ["", "-2", "33", "one"]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_int_list(bad, power2=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
