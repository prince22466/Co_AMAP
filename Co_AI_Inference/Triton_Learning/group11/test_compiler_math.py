#!/usr/bin/env python3
"""CPU-only regression tests. Does not require a GPU or a Triton installation."""
import unittest
import numpy as np
from compiler_reference import (
    assert_gpu_warp_size, base_arrays, fma_reference, licm_reference,
    live_range_reference, parse_tile_pairs, validate_tiling,
)
from compiler_lab_common import parse_ints


class CompilerReferenceTests(unittest.TestCase):
    def test_seeded_inputs_stable(self):
        a=base_arrays(13,2026)
        b=base_arrays(13,2026)
        for aa,bb in zip(a,b):
            np.testing.assert_array_equal(aa,bb)

    def test_fma_reference(self):
        x,a,b=base_arrays(11,1)
        z=fma_reference(x,a,b,13)
        expected=x.astype(np.float64)
        for _ in range(13): expected=expected*a.astype(np.float64)+b.astype(np.float64)
        np.testing.assert_array_equal(z,expected.astype(np.float32))

    def test_live_range_reference(self):
        x=np.arange(40,dtype=np.float32).reshape(8,5)*.01
        a=np.full(5,1,dtype=np.float32)
        b=np.full(5,.25,dtype=np.float32)
        z=live_range_reference(x,a,b,6)
        np.testing.assert_allclose(z,x[0]+1.5+x.sum(axis=0),atol=1e-6)

    def test_licm_reference(self):
        x=np.array([0.2,0.7],dtype=np.float32)
        a=np.array([.1,.2],dtype=np.float32)
        b=np.ones(2,dtype=np.float32)
        z=licm_reference(x,a,b,7)
        np.testing.assert_allclose(z,x.astype(np.float64)+7*np.exp(a.astype(np.float64)),atol=1e-6)

    def test_tile_parsing(self):
        self.assertEqual(parse_tile_pairs('16x64,32x32,16x64'),[(16,64),(32,32)])
        with self.assertRaises(ValueError): parse_tile_pairs('48x32')
        with self.assertRaises(ValueError): parse_tile_pairs('foo')
        with self.assertRaises(ValueError): validate_tiling(128,256,4)

    def test_int_parsing(self):
        self.assertEqual(parse_ints('1,2,4,2',label='unroll',powers_of_two=True),[1,2,4])
        with self.assertRaises(ValueError): parse_ints('3',label='unroll',powers_of_two=True)
        with self.assertRaises(ValueError): parse_ints('1,0',label='unroll')

    def test_warp_validation(self):
        self.assertEqual(assert_gpu_warp_size(64),64)
        self.assertEqual(assert_gpu_warp_size(32),32)
        with self.assertRaises(ValueError): assert_gpu_warp_size(48)

    def test_cpu_reference_nonmultiple_lengths(self):
        x,a,b=base_arrays(137,2026)
        self.assertEqual(fma_reference(x,a,b,9).shape,(137,))
        self.assertTrue(np.isfinite(fma_reference(x,a,b,9)).all())

if __name__=='__main__': unittest.main(verbosity=2)
