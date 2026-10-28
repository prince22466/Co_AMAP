"""CPU-only eligibility and shuffled physical-page address-equivalence tests."""

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "flag_gems" / "ops" / "iluvatar_grouped_paged_kv.py"
)
spec = importlib.util.spec_from_file_location("v4_grouped_paged_kv", MODULE_PATH)
lookup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lookup)


def enabled(**kw):
    args = {
        "is_paged": True,
        "max_seqlen_q": 2048,
        "max_seqlen_k": 16384,
        "block_size": 16,
        "block_n": 32,
        "is_dropout": False,
        "is_local": False,
        "mode": "grouped",
    }
    args.update(kw)
    return lookup.select_grouped_page_lookup(**args)


def test_opt_in_disabled_by_default(monkeypatch):
    monkeypatch.delenv(lookup.ENV_NAME, raising=False)
    assert not enabled(mode=None)
    monkeypatch.setenv(lookup.ENV_NAME, "grouped")
    assert enabled(mode=None)
    monkeypatch.setenv(lookup.ENV_NAME, "baseline")
    assert not enabled(mode=None)


@pytest.mark.parametrize("block_size,block_n", [(8, 32), (16, 32), (16, 64), (32, 32)])
def test_valid_aligned_pages(block_size, block_n):
    assert enabled(block_size=block_size, block_n=block_n)


@pytest.mark.parametrize("bad", [
    {"is_paged": False},
    {"max_seqlen_q": 1},
    {"max_seqlen_q": 127},
    {"max_seqlen_k": 4096},
    {"is_dropout": True},
    {"is_local": True},
    {"block_size": 0},
    {"block_size": 24},
    {"block_size": 64},
    {"block_n": 48},
])
def test_unsafe_eligibility_falls_back(bad):
    assert not enabled(**bad)


@pytest.mark.parametrize("value", ["true", "force", "auto", "128x32"])
def test_reject_invalid_modes(value):
    with pytest.raises(ValueError, match="Unsupported"):
        enabled(mode=value)


@pytest.mark.parametrize("block_size,block_n", [(8, 32), (16, 32), (16, 64), (32, 64)])
def test_shuffled_page_address_equivalence(block_size, block_n):
    # Non-monotonic physical pages guard against accidental identity mapping.
    table = [9, 0, 17, 3, 11, 4, 7, 2, 21, 6, 15, 1, 13, 8]
    page_stride = block_size * 2 * 128
    row_stride = 2 * 128
    for tile_start in (0, block_n, 2 * block_n):
        if tile_start + block_n > len(table) * block_size:
            continue
        grouped_pages = lookup.grouped_virtual_page_indices(
            tile_start, block_n, block_size
        )
        for lane, virtual_page in enumerate(grouped_pages):
            token = tile_start + lane
            original = (
                table[token // block_size] * page_stride
                + token % block_size * row_stride
            )
            grouped = (
                table[virtual_page] * page_stride
                + lane % block_size * row_stride
            )
            assert original == grouped


@pytest.mark.parametrize("start,size,n", [(8, 16, 32), (-16, 16, 32), (0, 16, 31)])
def test_reject_unaligned_page_or_tile(start, size, n):
    with pytest.raises(ValueError):
        lookup.grouped_virtual_page_indices(start, n, size)
