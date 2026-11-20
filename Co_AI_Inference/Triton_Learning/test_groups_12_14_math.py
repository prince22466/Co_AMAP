#!/usr/bin/env python3
"""CPU-only mathematical invariants (not proof of GPU kernel correctness)."""
import math
import unittest

try:
    import torch
except ImportError as exc:
    raise SystemExit("Requires torch; CPU build is sufficient") from exc


class LabMathTests(unittest.TestCase):
    def test_page_table_order_and_gqa(self):
        page, context, dim, q_heads, kv_heads = 4, 13, 8, 4, 2
        table = torch.tensor([3, 1, 0, 2])
        k = torch.arange(4 * kv_heads * page * dim).reshape(
            4, kv_heads, page, dim).float()
        ids = torch.arange(context)
        for qh in range(q_heads):
            kh = qh // (q_heads // kv_heads)
            mapped = k[table[ids // page], kh, ids % page]
            loop = torch.stack([k[int(table[t // page]), kh, t % page]
                                for t in range(context)])
            self.assertTrue(torch.equal(mapped, loop))

    def test_online_softmax_recurrence(self):
        torch.manual_seed(1)
        scores = torch.randn(37, dtype=torch.float64)
        values = torch.randn(37, 8, dtype=torch.float64)
        expected = torch.softmax(scores, 0) @ values
        maximum, denominator = -math.inf, 0.0
        weighted = torch.zeros(8, dtype=torch.float64)
        for start in range(0, 37, 4):
            s, v = scores[start:start+4], values[start:start+4]
            nxt = max(maximum, float(s.max()))
            previous = math.exp(maximum - nxt)
            weights = torch.exp(s - nxt)
            weighted = weighted * previous + weights @ v
            denominator = denominator * previous + float(weights.sum())
            maximum = nxt
        self.assertTrue(torch.allclose(weighted / denominator, expected,
                                       atol=1e-12, rtol=1e-12))

    def test_atomic_aggregation_bin_equivalence(self):
        n, block, bins = 1051, 128, 7
        ids = torch.arange(n) // block % bins
        per_lane = torch.bincount(ids, minlength=bins)
        per_cta = torch.zeros(bins, dtype=torch.int64)
        for start in range(0, n, block):
            per_cta[(start // block) % bins] += min(block, n - start)
        self.assertTrue(torch.equal(per_cta, per_lane))
        self.assertEqual(int(per_cta.sum()), n)

    def test_two_pass_reduction(self):
        values = torch.arange(1, 1030, dtype=torch.int64)
        partial = [int(values[i:i+128].sum()) for i in range(0, len(values), 128)]
        self.assertEqual(sum(partial), int(values.sum()))

    def test_transpose_reference(self):
        x = torch.arange(127*129).reshape(127, 129)
        self.assertTrue(torch.equal(x.T.contiguous().T.contiguous(), x))


if __name__ == "__main__":
    unittest.main()
