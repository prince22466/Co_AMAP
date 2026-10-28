"""CPU-only selector checks; the GPU kernel itself needs BI-V150 validation."""

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "flag_gems" / "ops" / "iluvatar_prefill_tiles.py"
)
spec = importlib.util.spec_from_file_location("iluvatar_prefill_tiles_v4", MODULE_PATH)
tiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiles)


def select(**overrides):
    kwargs = dict(
        max_seqlen_q=16384,
        max_seqlen_k=16384,
        head_size=128,
        is_dropout=False,
        is_local=False,
    )
    kwargs.update(overrides)
    return tiles.select_prefill_tiles(**kwargs)


def test_default_unchanged(monkeypatch):
    monkeypatch.delenv(tiles.ENV_NAME, raising=False)
    assert select() is None


@pytest.mark.parametrize("mode,expected", list(tiles.TILE_CANDIDATES.items()))
def test_explicit_candidates(mode, expected):
    assert select(mode=mode) == expected


def test_non_prefill_and_special_attention_unchanged():
    assert select(mode="128x64", max_seqlen_q=4096) is None
    assert select(mode="128x64", max_seqlen_k=4096) is None
    assert select(mode="128x64", is_dropout=True) is None
    assert select(mode="128x64", is_local=True) is None


@pytest.mark.parametrize(
    "head_size,query_len,expected",
    [(64, 16384, (128, 64)), (128, 16384, (64, 64)),
     (256, 16384, (64, 32)), (64, 8192, (64, 64))],
)
def test_shape_dependent_candidate(head_size, query_len, expected):
    assert select(mode="auto", head_size=head_size, max_seqlen_q=query_len) == expected


def test_mode_environment(monkeypatch):
    monkeypatch.setenv(tiles.ENV_NAME, "128x64")
    assert select() == (128, 64)


def test_invalid_mode_fails_closed():
    with pytest.raises(ValueError, match="Unsupported"):
        select(mode="256x128")


def test_register_pressure_guard():
    with pytest.raises(ValueError, match="accumulator guard"):
        select(mode="256x32", head_size=256)

def test_unwrap_libentry_launch_return():
    class CompiledKernel:
        n_regs = 64
        n_spills = 0

    compiled = CompiledKernel()
    assert tiles.unwrap_libentry_kernel((compiled, {"BLOCK_M": 128})) is compiled
    assert tiles.unwrap_libentry_kernel(compiled) is compiled


def page_lookup(**overrides):
    params = dict(
        is_paged=True,
        max_seqlen_q=2048,
        max_seqlen_k=16384,
        block_size=16,
        block_n=32,
        is_dropout=False,
        is_local=False,
        mode="grouped",
    )
    params.update(overrides)
    return tiles.select_grouped_page_lookup(**params)


def test_grouped_lookup_default_off(monkeypatch):
    monkeypatch.delenv(tiles.PAGED_KV_ENV, raising=False)
    assert not page_lookup(mode=None)
    monkeypatch.setenv(tiles.PAGED_KV_ENV, "grouped")
    assert page_lookup(mode=None)


@pytest.mark.parametrize("page_size,kv_tile", [(8, 32), (16, 32), (16, 64), (32, 32)])
def test_grouped_page_sizes(page_size, kv_tile):
    assert page_lookup(block_size=page_size, block_n=kv_tile)


@pytest.mark.parametrize(
    "override",
    [
        {"is_paged": False},
        {"max_seqlen_q": 1},
        {"max_seqlen_q": 127},
        {"max_seqlen_k": 4096},
        {"block_size": 24},
        {"block_size": 64},
        {"block_size": 0},
        {"is_dropout": True},
        {"is_local": True},
    ],
)
def test_grouped_lookup_safety_gates(override):
    assert not page_lookup(**override)


def test_grouped_lookup_invalid_mode():
    with pytest.raises(ValueError, match="Unsupported"):
        page_lookup(mode="force")


@pytest.mark.parametrize("page_size,kv_tile", [(8, 32), (16, 32), (16, 64), (32, 64)])
def test_grouped_offset_matches_per_token_reference(page_size, kv_tile):
    # Non-monotonic physical page table exposes incorrect page-index arithmetic.
    physical_pages = [9, 0, 17, 3, 11, 4, 7, 2, 21, 6, 15, 1, 13, 8]
    k_page_stride = page_size * 2 * 128
    k_row_stride = 2 * 128
    for tile_start in (0, kv_tile, 2 * kv_tile):
        if tile_start + kv_tile > len(physical_pages) * page_size:
            continue
        grouped = tiles.grouped_page_indices(tile_start, kv_tile, page_size)
        reference = [(tile_start + lane) // page_size for lane in range(kv_tile)]
        assert grouped == reference
        optimized_offsets = [
            physical_pages[virtual_page] * k_page_stride
            + (lane % page_size) * k_row_stride
            for lane, virtual_page in enumerate(grouped)
        ]
        reference_offsets = [
            physical_pages[virtual_page] * k_page_stride
            + ((tile_start + lane) % page_size) * k_row_stride
            for lane, virtual_page in enumerate(reference)
        ]
        assert optimized_offsets == reference_offsets


def test_grouped_lookup_rejects_misalignment():
    with pytest.raises(ValueError, match="boundary"):
        tiles.grouped_page_indices(8, 32, 16)
