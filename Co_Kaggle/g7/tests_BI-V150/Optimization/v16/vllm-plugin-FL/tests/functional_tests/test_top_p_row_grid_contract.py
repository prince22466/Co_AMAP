# SPDX-License-Identifier: Apache-2.0
"""CPU-only launch contract tests; no vLLM or Triton installation required."""

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def wrapper_module():
    source = Path(__file__).resolve().parents[2] / "vllm_fl/ops/topk_topp.py"
    spec = importlib.util.spec_from_file_location("row_grid_contract", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTensor:
    def __init__(self, shape, *, dtype=torch.float32, strides=None, device=None):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = device or torch.device("cuda:0")
        self.strides = strides or ((shape[1], 1) if self.ndim == 2 else (1,))

    def stride(self, dim):
        return self.strides[dim]

    def to(self, dtype):
        return FakeTensor(self.shape, dtype=dtype, device=self.device)

    def new_empty(self, shape):
        return FakeTensor(shape, device=self.device, dtype=self.dtype)

    def new_tensor(self, values):
        return self.new_empty((len(values),))


class KernelSpy:
    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append((grid, args, kwargs))
        return launch


@pytest.fixture
def upstream():
    return SimpleNamespace(
        apply_top_k_top_p_triton=Mock(return_value="original"),
        num_compute_units=Mock(return_value=16),
        _TRITON_BUFFER_CACHE={},
        _TRITON_TABLE_CACHE={},
        _NORMAL_CDF_TO_SIGMA_TABLE=[1.0],
        _PERCENTILE_TO_STD_TABLE=[2.0],
        _topk_topp_kernel=KernelSpy(),
    )


@pytest.mark.parametrize("rows, programs", [(17, 17), (32, 32), (64, 64), (65, 64)])
def test_expanded_launch_contract(wrapper_module, upstream, rows, programs):
    wrapper = wrapper_module._build_wrapper(upstream)
    x = FakeTensor((rows, 8193), strides=(8200, 1))
    k = FakeTensor((rows,), dtype=torch.int64)
    p = FakeTensor((rows,))
    for _ in range(2):
        assert wrapper(x, k, p, -123.0) is x
    launches = upstream._topk_topp_kernel.launches
    assert len(launches) == 2
    for grid, args, kwargs in launches:
        assert grid == (programs,)
        assert args[0] is x and args[1] == 8200
        assert args[2].shape == (programs, 8193)
        assert args[5].dtype == torch.int32
        assert args[6].dtype == torch.float32
        assert kwargs == dict(
            BATCH_SIZE=rows, MASK_VALUE=-123.0, VOCAB_SIZE=8193,
            BLOCK_SIZE=8192, BLOCK_SIZE_TRUNC=4096,
            TOPK_ENABLED=True, TOPP_ENABLED=True,
        )
    assert launches[0][1][2] is not launches[1][1][2]
    assert upstream._TRITON_BUFFER_CACHE == {}
    assert len(upstream._TRITON_TABLE_CACHE) == 1
    upstream.apply_top_k_top_p_triton.assert_not_called()


@pytest.mark.parametrize("case", ["small", "top-k", "overlap", "columns", "p-stride", "k-stride", "cpu", "dtype"])
def test_original_fallback(wrapper_module, upstream, case):
    rows = 16 if case == "small" else 64
    strides = {"overlap": (0, 1), "columns": (8193 * 2, 2)}.get(case)
    x = FakeTensor(
        (rows, 8193), strides=strides,
        dtype=torch.float16 if case == "dtype" else torch.float32,
        device=torch.device("cpu") if case == "cpu" else None,
    )
    p = None if case == "top-k" else FakeTensor(
        (rows,), strides=(2,) if case == "p-stride" else None, device=x.device,
    )
    k = FakeTensor((rows,), strides=(2,) if case == "k-stride" else None,
                   device=x.device, dtype=torch.int32)
    wrapper = wrapper_module._build_wrapper(upstream)
    assert wrapper(x, k, p) == "original"
    upstream.apply_top_k_top_p_triton.assert_called_once_with(x, k, p, float("-inf"))
    assert not upstream._topk_topp_kernel.launches


def test_install_source_guard(wrapper_module, upstream, monkeypatch, tmp_path):
    source = tmp_path / "topk_topp_triton.py"
    source.write_bytes(b"tested source")
    upstream.__file__ = str(source)
    original = upstream.apply_top_k_top_p_triton
    sampler = SimpleNamespace(apply_top_k_top_p_triton=original)
    for name in ("vllm", "vllm.v1", "vllm.v1.sample", "vllm.v1.sample.ops"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    ops = sys.modules["vllm.v1.sample.ops"]
    ops.topk_topp_sampler = sampler
    ops.topk_topp_triton = upstream
    assert not wrapper_module.install_top_p_row_grid()
    assert sampler.apply_top_k_top_p_triton is original

    monkeypatch.setattr(wrapper_module, "_SUPPORTED_SOURCE_SHA256",
                        hashlib.sha256(source.read_bytes()).hexdigest())
    replacement = Mock()
    sampler.apply_top_k_top_p_triton = replacement
    assert not wrapper_module.install_top_p_row_grid()
    assert sampler.apply_top_k_top_p_triton is replacement

    sampler.apply_top_k_top_p_triton = original
    assert wrapper_module.install_top_p_row_grid()
    selected = sampler.apply_top_k_top_p_triton
    assert selected._fl_row_grid_original is original
    assert wrapper_module.install_top_p_row_grid()
    assert sampler.apply_top_k_top_p_triton is selected
    assert upstream.apply_top_k_top_p_triton is original
