#!/usr/bin/env python3
"""CPU-only sanity checks for Group4's reference/index calculations.

Can run without Triton or a GPU. These are not substitutes for on-device
numerical validation and compiled-ISA inspection.
"""
import numpy as np
from memory_lab_common import (
    chase_reference, positive_ints, powers_of_two, sample_positions,
    stream_reference, working_sets_mib,
)


def main():
    assert powers_of_two('4,8,4', 'test') == [4, 8]
    assert positive_ints('1,17,257', 'stride') == [1, 17, 257]
    assert [entries for _, entries in working_sets_mib('0.25,1')] == [65536, 262144]

    ws = 64
    n = 19
    reps = 9
    pool = np.arange(ws, dtype=np.float32) * np.float32(0.125)
    places = sample_positions(n)
    for stride in (1, 3, 17):
        got = stream_reference(pool, n, reps, stride, places, ws)
        brute = np.array([sum(float(pool[((int(p) * stride + t*n) & (ws-1))])
                              for t in range(reps)) for p in places], dtype=np.float32)
        np.testing.assert_allclose(got, brute, rtol=1e-6, atol=1e-5)

    rng = np.random.default_rng(77)
    table = rng.permutation(128).astype(np.int32)
    for chains in (1, 2, 4, 8, 16):
        total = 32
        got = chase_reference(table, chains, total, places)
        expected = []
        for pos in places:
            total_val = 0
            for chain in range(chains):
                index = (int(pos) + chain * 65537) & 127
                for _ in range(total // chains):
                    index = int(table[index])
                total_val += index
            expected.append(total_val)
        np.testing.assert_array_equal(got, np.array(expected, dtype=np.int32))
    print('PASS: M01/M04 stream indexing and M02/M03 pointer-chain references')


if __name__ == '__main__':
    main()
