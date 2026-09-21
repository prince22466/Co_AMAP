#!/usr/bin/env python3
"""Export a trained v21 checkpoint to a quantized Kaggriculture main.py.

The exported submission uses kaggriculture-sub_v20.ipynb as the policy template,
but replaces its embedded residual-Q tensor with the selected v21 checkpoint's
quantized deployment weights.

Storage formats:
- fp16: 2 bytes/parameter, decoded by struct.unpack("<6721e", ...)
- fp8_e4m3fn: 1 byte/parameter, decoded by a small stdlib E4M3FN decoder

Arithmetic after decoding is still Python float arithmetic; the model parameters
themselves are restricted to the selected quantized representable values.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import torch

from train_v21_static_history import (
    DEFAULT_V20_SUBMISSION,
    GLOBAL_FEATURE_NAMES,
    TASK_FEATURE_NAMES,
    QuantizedResidualQ,
    _extract_notebook_main,
)
from quantization import flattened_submission_weights

HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = HERE / "runs" / "static_v20_history" / "checkpoints" / "latest.pt"
DEFAULT_OUTPUT_DIR = HERE / "runs" / "static_v20_history" / "submission"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _assignment_node(source: str, name: str) -> ast.Assign:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return node
    raise ValueError(f"submission main.py does not define {name}")


def _wrap_b64(data: bytes, width: int = 120) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    chunks = [encoded[i : i + width] for i in range(0, len(encoded), width)]
    return "_Q_WEIGHTS_B64=(\n" + "".join(f'    "{chunk}"\n' for chunk in chunks) + ")"


def _decode_fp8_code(code: int) -> float:
    sign = -1.0 if code & 0x80 else 1.0
    exponent = (code >> 3) & 0x0F
    mantissa = code & 0x07
    if exponent == 0x0F and mantissa == 0x07:
        raise ValueError("NaN encoding is not a finite E4M3FN weight")
    if exponent == 0:
        value = mantissa * (2.0 ** -9)
    else:
        value = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
    return sign * value


def _encode_fp8_e4m3fn(values: np.ndarray) -> bytes:
    codebook: dict[float, int] = {}
    for code in range(256):
        if (code & 0x7F) == 0x7F:
            continue
        value = _decode_fp8_code(code)
        # Prefer positive zero for the duplicate +/-0 representation.
        if value == 0.0 and code & 0x80:
            continue
        codebook[value] = code

    output = bytearray()
    for value in values.astype(np.float32, copy=False):
        key = float(value)
        code = codebook.get(key)
        if code is None:
            # Values should already come from the exact software E4M3FN quantizer.
            nearest_value, code = min(
                ((abs(key - representable), byte) for representable, byte in codebook.items()),
                key=lambda pair: pair[0],
            )
            if nearest_value > 0.0:
                raise ValueError(f"non-E4M3FN deployment value encountered: {key}")
        output.append(code)
    return bytes(output)


def _replace_quantized_weights(source: str, blob: bytes, quantization: str) -> str:
    weights_node = _assignment_node(source, "_Q_WEIGHTS_B64")
    all_node = _assignment_node(source, "_Q_ALL")
    lines = source.splitlines()

    start = weights_node.lineno - 1
    end = all_node.end_lineno

    assignment = _wrap_b64(blob)
    if quantization == "fp16":
        decoder = '_Q_ALL=struct.unpack("<6721e",base64.b64decode(_Q_WEIGHTS_B64))'
    elif quantization == "fp8_e4m3fn":
        decoder = """def _q_decode_fp8_e4m3fn(blob):
    out=[]
    for code in blob:
        sign=-1.0 if code&0x80 else 1.0
        exponent=(code>>3)&0x0F
        mantissa=code&0x07
        if exponent==0x0F and mantissa==0x07:
            raise ValueError("invalid E4M3FN NaN weight")
        if exponent==0:
            value=mantissa*(2.0**-9)
        else:
            value=(1.0+mantissa/8.0)*(2.0**(exponent-7))
        out.append(sign*value)
    return tuple(out)

_Q_ALL=_q_decode_fp8_e4m3fn(base64.b64decode(_Q_WEIGHTS_B64))"""
    else:
        raise ValueError(f"unsupported quantization: {quantization}")

    replacement = (assignment + "\n\n" + decoder).splitlines()
    updated = lines[:start] + replacement + lines[end:]

    if updated and updated[0].startswith('"""Kaggriculture v20'):
        updated[0] = updated[0].replace("Kaggriculture v20", "Kaggriculture v21", 1)
    return "\n".join(updated) + "\n"


def _load_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("algorithm") != "v21_static_residual_double_dqn":
        raise ValueError(
            f"{path}: expected v21_static_residual_double_dqn, "
            f"got {payload.get('algorithm')!r}"
        )
    quantization = payload.get("quantization", "fp16")
    model = QuantizedResidualQ(
        len(TASK_FEATURE_NAMES),
        len(GLOBAL_FEATURE_NAMES),
        64,
        quantization,
    ).to(device)
    model.load_state_dict(payload["online_state_dict"])
    model.eval()
    return model, payload, quantization


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--v20-submission", type=Path, default=DEFAULT_V20_SUBMISSION)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--device", default="cpu")
    return p


def main():
    args = build_parser().parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    v20_submission = args.v20_submission.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    if not v20_submission.is_file():
        raise SystemExit(f"v20 submission not found: {v20_submission}")

    device = torch.device(args.device)
    model, payload, quantization = _load_checkpoint(checkpoint, device)
    flat = flattened_submission_weights(model, quantization).numpy()

    if quantization == "fp16":
        blob = flat.astype("<f2").tobytes()
        bytes_per_parameter = 2
    elif quantization == "fp8_e4m3fn":
        blob = _encode_fp8_e4m3fn(flat)
        bytes_per_parameter = 1
    else:
        raise ValueError(f"unsupported quantization: {quantization}")

    expected_bytes = 6721 * bytes_per_parameter
    if len(blob) != expected_bytes:
        raise ValueError(f"expected {expected_bytes} model bytes, got {len(blob)}")

    template = _extract_notebook_main(v20_submission)
    main_source = _replace_quantized_weights(template, blob, quantization)
    main_bytes = main_source.encode("utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = output_dir / "main.py"
    main_path.write_bytes(main_bytes)

    archive_path = output_dir / "submission.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(main_path, arcname="main.py", recursive=False)

    metadata = {
        "algorithm": payload.get("algorithm"),
        "checkpoint": str(checkpoint),
        "checkpoint_update": payload.get("update"),
        "quantization": quantization,
        "parameter_count": 6721,
        "bytes_per_parameter": bytes_per_parameter,
        "raw_quantized_model_bytes": len(blob),
        "model_blob_sha256": _sha256_bytes(blob),
        "main_py_sha256": _sha256_bytes(main_bytes),
        "archive": str(archive_path),
        "template": str(v20_submission),
        "note": (
            "Weights/biases are stored in the selected quantized format; "
            "stdlib submission arithmetic after decoding uses Python floats."
        ),
    }
    (output_dir / "export_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
