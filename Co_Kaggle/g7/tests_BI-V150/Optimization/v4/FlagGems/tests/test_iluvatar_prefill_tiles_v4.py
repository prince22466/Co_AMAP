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
