"""Quantization helpers for v21 residual-Q training and export.

Default FP16 mode is true end-to-end FP16 for the trainable network:
parameters, activations, Q values, TD targets, gradients and Adam moments all
use float16. There is no FP32 master-weight copy.

The optional FP8 mode remains quantization-aware: parameters use FP32 storage
while each forward pass uses software E4M3FN-quantized weights.

Supported formats:
- fp16: true IEEE-754 binary16 training and deployment
- fp8_e4m3fn: software E4M3FN-style finite FP8 weight QAT
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
    """Tree-free v21 Q network with true FP16 or FP8-aware execution."""

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
        if quantization == "fp16":
            self.half()

    @property
    def compute_dtype(self) -> torch.dtype:
        return torch.float16 if self.quantization == "fp16" else torch.float32

    def residual(self, states: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if self.quantization == "fp16":
            x = torch.cat(
                [
                    states.to(dtype=torch.float16),
                    candidates.to(dtype=torch.float16),
                ],
                dim=-1,
            )
            return self.net(x).squeeze(-1)

        x = torch.cat([states, candidates], dim=-1).to(torch.float32)
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

    def q_values(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        prior: torch.Tensor | None = None,
        prior_scale: float = 0.0,
    ) -> torch.Tensor:
        """Return the neural Q value only.

        v21 is tree-free: the old learned_task_score/tree prior is ignored.
        The prior arguments remain only for compatibility with the shared
        replay/Double-DQN call sites.
        """
        dtype = self.compute_dtype
        state = state.to(dtype=dtype)
        candidates = candidates.to(dtype=dtype)
        if state.ndim == 1:
            state = state.unsqueeze(0).expand(candidates.shape[0], -1)
        elif state.shape[0] == 1 and candidates.shape[0] != 1:
            state = state.expand(candidates.shape[0], -1)
        return self.residual(state, candidates)


def quantized_state_dict(model: ResidualQ, quantization: str) -> dict[str, torch.Tensor]:
    """Return a deployment state dict in the selected low-precision format."""
    if quantization == "fp16":
        return {
            name: tensor.detach().cpu().to(torch.float16)
            for name, tensor in model.state_dict().items()
        }
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


class PureFP16Adam(torch.optim.Optimizer):
    """Adam whose trainable state is genuinely FP16.

    Parameter tensors, gradients, exp_avg and exp_avg_sq are float16. The step
    count is a Python integer (bookkeeping, not floating-point model state).
    A representable FP16 epsilon is used by default because 1e-8 underflows.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-5,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-4,
    ):
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if eps <= 0:
            raise ValueError("eps must be positive")
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.dtype != torch.float16:
                    raise TypeError(
                        f"PureFP16Adam requires float16 parameters, got {param.dtype}"
                    )
                grad = param.grad
                if grad.dtype != torch.float16:
                    raise TypeError(
                        f"PureFP16Adam requires float16 gradients, got {grad.dtype}"
                    )
                if grad.is_sparse:
                    raise RuntimeError("PureFP16Adam does not support sparse gradients")

                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1
                denom = exp_avg_sq.sqrt().div_(bias_correction2 ** 0.5).add_(eps)
                param.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
