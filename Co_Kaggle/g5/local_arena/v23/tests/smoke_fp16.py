#!/usr/bin/env python3
"""No-API smoke test for v23 FP16 enforcement."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from replay.runner import (
    _audit_fp16_source,
    _numpy_fp16_candidate_defaults,
    _set_fp16_defaults,
)


def main() -> int:
    info = _set_fp16_defaults()
    assert torch.get_default_dtype() == torch.float16, info

    with tempfile.TemporaryDirectory(prefix="v23_fp16_") as tmp:
        tmp = Path(tmp)

        good = tmp / "good.py"
        good.write_text(
            "import torch\n"
            "W = torch.tensor([1.0], dtype=torch.float16)\n"
            "def agent(obs): return {'farmer':['PASS'],'hands':[],'market':[]}\n",
            encoding="utf-8",
        )
        assert _audit_fp16_source(good)["ok"]

        bad = tmp / "bad.py"
        bad.write_text(
            "import torch\n"
            "W = torch.tensor([1.0], dtype=torch.float32)\n"
            "def agent(obs): return {'farmer':['PASS'],'hands':[],'market':[]}\n",
            encoding="utf-8",
        )
        audit = _audit_fp16_source(bad)
        assert not audit["ok"]
        assert audit["violations"]

    original_float = np.array([1.0])
    original_int = np.array([1])
    assert original_float.dtype != np.float16
    assert np.issubdtype(original_int.dtype, np.integer)

    with _numpy_fp16_candidate_defaults():
        float_arr = np.array([1.0])
        int_arr = np.array([1])
        zeros = np.zeros(4)
        assert float_arr.dtype == np.float16
        assert zeros.dtype == np.float16
        assert np.issubdtype(int_arr.dtype, np.integer)

    print("v23 FP16 enforcement smoke test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
