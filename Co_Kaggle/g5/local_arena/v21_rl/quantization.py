"""Quantization helpers for v21 residual-Q training and export.

Training uses FP32 master parameters for optimizer stability, but every forward
pass fake-quantizes weights/biases to the selected deployment format. This is
weight-only QAT: states and intermediate activations remain FP32.

Supported formats:
- fp16: IEEE-754 binary16
- fp8_e4m3fn: software E4M3FN-style finite FP8 (max magnitude 448)
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from train_v20_q_history import ResidualQ

QUANTIZATION_CHOICES = ("fp16", "fp8_e4m3fn")


def _ste(original: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    return original + (quantized - original).detach()


def quantize_fp16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.float16).to(torch.float32)


def quantize_fp8_e4m3fn(tensor: torch.Tensor) -> torch.Tensor:
    """Software E4M3FN quantizer returning representable values in FP32.

    E4M3FN uses 1 sign bit, 4 exponent bits and 3 mantissa bits. This software
    path includes subnormals with step 2^-9 and finite values up to 448.
    """
    x = tensor.to(torch.float32)
    sign = torch.sign(x)
    a = x.abs()
    max_finite = 448.0
    min_normal = 2.0 ** -6
    subnormal_step = 2.0 ** -9

    clipped = torch.clamp(a, max=max_finite)
    sub = torch.round(clipped / subnormal_step) * subnormal_step

    safe = torch.clamp(clipped, min=min_normal)
    exponent = torch.floor(torch.log2(safe))
    exponent = torch.clamp(exponent, min=-6.0, max=8.0)
    step = torch.pow(
        torch.tensor(2.0, device=x.device, dtype=torch.float32),
        exponent - 3.0,
    )
    normal = torch.round(clipped / step) * step
    normal = torch.clamp(normal, max=max_finite)

    q = torch.where(clipped < min_normal, sub, normal)
    q = torch.where(a == 0, torch.zeros_like(q), q)
    return sign * q


def quantize_tensor(tensor: torch.Tensor, quantization: str) -> torch.Tensor:
    if quantization == "fp16":
        return quantize_fp16(tensor)
    if quantization == "fp8_e4m3fn":
        return quantize_fp8_e4m3fn(tensor)
    raise ValueError(f"unsupported quantization: {quantization}")


def fake_quantize_tensor(tensor: torch.Tensor, quantization: str) -> torch.Tensor:
    return _ste(tensor, quantize_tensor(tensor, quantization))


class QuantizedResidualQ(ResidualQ):
    """v20 ResidualQ with weight/bias fake quantization on every forward pass."""

    def __init__(
        self,
        task_dim: int,
        state_dim: int,
        hidden: int = 64,
        quantization: str = "fp16",
    ):
        if quantization not in QUANTIZATION_CHOICES:
            raise ValueError(
                f"quantization must be one of {QUANTIZATION_CHOICES}, got {quantization!r}"
            )
        super().__init__(task_dim, state_dim, hidden)
        self.quantization = quantization

    def residual(self, states: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, candidates], dim=-1)
        linear0: nn.Linear = self.net[0]
        linear2: nn.Linear = self.net[2]
        linear4: nn.Linear = self.net[4]

        x = F.linear(
            x,
            fake_quantize_tensor(linear0.weight, self.quantization),
            fake_quantize_tensor(linear0.bias, self.quantization),
        )
        x = torch.tanh(x)
        x = F.linear(
            x,
            fake_quantize_tensor(linear2.weight, self.quantization),
            fake_quantize_tensor(linear2.bias, self.quantization),
        )
        x = torch.tanh(x)
        x = F.linear(
            x,
            fake_quantize_tensor(linear4.weight, self.quantization),
            fake_quantize_tensor(linear4.bias, self.quantization),
        )
        return x.squeeze(-1)


def quantized_state_dict(model: ResidualQ, quantization: str) -> dict[str, torch.Tensor]:
    """Return a deployment state dict containing only quantized FP32 values."""
    return {
        name: quantize_tensor(tensor.detach().cpu(), quantization)
        for name, tensor in model.state_dict().items()
    }


def flattened_submission_weights(
    model: ResidualQ,
    quantization: str,
) -> torch.Tensor:
    """Flatten parameters in the exact order used by the v20 submission."""
    state = quantized_state_dict(model, quantization)
    parts = [
        state["net.0.weight"].reshape(-1),
        state["net.0.bias"].reshape(-1),
        state["net.2.weight"].reshape(-1),
        state["net.2.bias"].reshape(-1),
        state["net.4.weight"].reshape(-1),
        state["net.4.bias"].reshape(-1),
    ]
    flat = torch.cat(parts).to(torch.float32)
    if flat.numel() != 6721:
        raise ValueError(f"expected 6721 residual-Q parameters, got {flat.numel()}")
    return flat
