import torch
from typing import Optional


def sanitize_gradients(grad_dict: dict, name: str = "") -> dict:
    """Remove NaN/Inf gradients and clip extreme values."""
    sanitized = {}
    for key, tensor in grad_dict.items():
        if tensor is None:
            continue
        mask = torch.isfinite(tensor)
        if not mask.all():
            print(f"WARNING: NaN/Inf detected in {name}/{key}, masking {((~mask).sum().item())} values")
            tensor = torch.where(mask, tensor, torch.zeros_like(tensor))
        tensor = tensor.clamp(-1e3, 1e3)
        sanitized[key] = tensor
    return sanitized


def aggregate_gradients(
    grad_file_paths: list[str],
    weights: Optional[list[float]] = None,
    max_grad_norm: Optional[float] = 1.0,
) -> dict:
    if not grad_file_paths:
        return {}

    all_grads = []
    for path in grad_file_paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(data, dict) and "gradients" in data:
            raw = data["gradients"]
        elif isinstance(data, dict) and "model_state_dict" in data:
            raw = data["model_state_dict"]
        else:
            raw = data
        raw = sanitize_gradients(raw, path)
        all_grads.append(raw)

    n = len(all_grads)
    if n == 0:
        return {}

    w = weights if weights is not None else [1.0 / n] * n

    token_counts = []
    for path in grad_file_paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        count = data.get("token_count", 1024) if isinstance(data, dict) else 1024
        token_counts.append(count)

    total_tokens = sum(token_counts)
    if total_tokens > 0:
        w = [c / total_tokens for c in token_counts]

    averaged = {}
    for key in all_grads[0]:
        stacked = torch.stack([g[key].float() for g in all_grads])
        weight_tensor = torch.tensor(w).view(-1, *([1] * (stacked.dim() - 1)))
        averaged[key] = (stacked * weight_tensor).sum(dim=0)

    if max_grad_norm is not None:
        total_norm = 0.0
        for grad in averaged.values():
            total_norm += grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        if total_norm > max_grad_norm:
            scale = max_grad_norm / (total_norm + 1e-8)
            for key in averaged:
                averaged[key] = averaged[key] * scale

    return averaged
