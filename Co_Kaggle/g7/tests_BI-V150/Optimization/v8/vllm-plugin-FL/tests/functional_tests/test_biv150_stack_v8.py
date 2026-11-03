"""v8 selector and phase-routing regression tests (GPU tests run separately)."""
import pytest

from vllm_fl.dispatch.backends.vendor.iluvatar.impl.attention_stack_v8 import (
    get_modes,
    select_phase,
)


@pytest.mark.parametrize("value", ["none", "v1", "v2", "v4"])
def test_prefill_modes(monkeypatch, value):
    monkeypatch.setenv("VLLM_FL_BIV150_V8_PREFILL", value)
    monkeypatch.setenv("VLLM_FL_BIV150_V8_DECODE", "none")
    assert get_modes() == (value, "none")


def test_default_modes(monkeypatch):
    monkeypatch.delenv("VLLM_FL_BIV150_V8_PREFILL", raising=False)
    monkeypatch.delenv("VLLM_FL_BIV150_V8_DECODE", raising=False)
    assert get_modes() == ("v1", "none")


def test_invalid_modes(monkeypatch):
    monkeypatch.setenv("VLLM_FL_BIV150_V8_PREFILL", "v1+v2")
    with pytest.raises(ValueError):
        get_modes()
    monkeypatch.setenv("VLLM_FL_BIV150_V8_PREFILL", "v1")
    monkeypatch.setenv("VLLM_FL_BIV150_V8_DECODE", "unknown")
    with pytest.raises(ValueError):
        get_modes()


def test_pure_decode_vs_mixed_routing():
    assert select_phase(1, "v1", "v5") == "v5"
    assert select_phase(2048, "v1", "v5") == "v1"
    assert select_phase(128, "v4", "v5") == "v4"
    assert select_phase(1, "none", "v5") == "v5"
    assert select_phase(4096, "none", "v5") == "none"
