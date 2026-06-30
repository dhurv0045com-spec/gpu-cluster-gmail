"""Compact, versioned gradient/delta payloads for the An-Ra workers.

The top-k selection is exact but chunked by parameter. This avoids constructing
another 499M-element flattened tensor beside the model, which is the difference
between fitting on a Colab worker and an avoidable out-of-memory failure.
"""

import os
import re
from collections.abc import Mapping

import numpy as np
import torch


def _gradient_and_size(value):
    if isinstance(value, torch.nn.Parameter):
        return value.grad, value.numel()
    if value is None:
        return None, 0
    return value, value.numel()


def sparsify_gradients(
    named_params: Mapping,
    token_count: int,
    worker_id: str,
    step: int,
    model_version: int,
    topk_fraction: float = 0.01,
) -> dict:
    """Return the exact global top-k gradients without a full flat copy."""
    if not 0 < topk_fraction <= 1:
        raise ValueError("topk_fraction must be in (0, 1]")
    if token_count < 1:
        raise ValueError("token_count must be positive")

    entries = []
    total = 0
    for name, value in named_params.items():
        gradient, size = _gradient_and_size(value)
        entries.append((name, gradient, size, total))
        total += size
    if total == 0:
        raise ValueError("No model parameters were provided")
    if total > np.iinfo(np.int32).max:
        raise ValueError("Sparse payload format supports at most int32-addressable models")

    k = max(1, int(total * topk_fraction))
    candidate_values = torch.empty(0, dtype=torch.float32)
    candidate_indices = torch.empty(0, dtype=torch.int64)

    for _, gradient, size, offset in entries:
        if gradient is None or size == 0:
            continue
        flat = gradient.detach().reshape(-1).float().cpu()
        # A local value outside this parameter's top-k cannot be in the global
        # top-k, so retaining only these candidates is exact.
        local_k = min(k, flat.numel())
        _, local_indices = torch.topk(flat.abs(), local_k, sorted=False)
        values = flat[local_indices]
        indices = local_indices.to(torch.int64) + offset

        candidate_values = torch.cat((candidate_values, values))
        candidate_indices = torch.cat((candidate_indices, indices))
        if candidate_values.numel() > k:
            _, keep = torch.topk(candidate_values.abs(), k, sorted=False)
            candidate_values = candidate_values[keep]
            candidate_indices = candidate_indices[keep]

    if candidate_values.numel() == 0:
        raise ValueError("No gradients are available to sparsify")

    order = torch.argsort(candidate_indices)
    return {
        "indices": candidate_indices[order].numpy().astype(np.int32, copy=False),
        "values": candidate_values[order].numpy().astype(np.float16, copy=False),
        "total_params": np.array([total], dtype=np.int64),
        "token_count": np.array([token_count], dtype=np.int64),
        "worker_id": np.array([worker_id]),
        "step": np.array([step], dtype=np.int64),
        "model_version": np.array([model_version], dtype=np.int64),
        # This schema lets the model-free coordinator periodically fold deltas
        # into a checkpoint without guessing whether state_dict buffers count
        # toward named_parameters offsets.
        "parameter_names": np.asarray([name for name, _, _, _ in entries]),
        "parameter_sizes": np.asarray([size for _, _, size, _ in entries], dtype=np.int64),
    }


def save_sparse_gradient(sparse_gradient: dict, path: str) -> None:
    """Write atomically so Drive sync never exposes a half-written archive."""
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp.npz"
    np.savez_compressed(temporary, **sparse_gradient)
    os.replace(temporary, path)


def load_sparse_gradient(path: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "indices",
            "values",
            "total_params",
            "token_count",
            "worker_id",
            "step",
            "model_version",
            "parameter_names",
            "parameter_sizes",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Sparse payload is missing fields: {sorted(missing)}")
        indices = data["indices"].astype(np.int32, copy=True)
        values = data["values"].astype(np.float32, copy=True)
        if indices.shape != values.shape or indices.ndim != 1:
            raise ValueError("Sparse indices and values must be matching one-dimensional arrays")
        return {
            "indices": indices,
            "values": values,
            "total_params": int(data["total_params"][0]),
            "token_count": int(data["token_count"][0]),
            "worker_id": str(data["worker_id"][0]),
            "step": int(data["step"][0]),
            "model_version": int(data["model_version"][0]),
            "parameter_names": data["parameter_names"].astype(str).tolist(),
            "parameter_sizes": data["parameter_sizes"].astype(np.int64).tolist(),
            "learning_rate": float(data["learning_rate"][0]) if "learning_rate" in data.files else 1.0,
        }


def apply_sparse_delta_to_model(model, flat_indices: np.ndarray, flat_values: np.ndarray) -> None:
    """Apply a sparse optimizer delta in place, using O(entries) index slicing."""
    indices = np.asarray(flat_indices, dtype=np.int64)
    values = np.asarray(flat_values, dtype=np.float32)
    if indices.ndim != 1 or values.ndim != 1 or indices.shape != values.shape:
        raise ValueError("Sparse indices and values must be matching one-dimensional arrays")
    if indices.size == 0:
        return
    if not np.all(np.isfinite(values)):
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(indices[1:] < indices[:-1]):
        order = np.argsort(indices, kind="stable")
        indices = indices[order]
        values = values[order]

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if indices[0] < 0 or indices[-1] >= total_parameters:
        raise ValueError("Sparse delta contains an index outside the model")

    offset = 0
    with torch.no_grad():
        for _, parameter in model.named_parameters():
            end = offset + parameter.numel()
            left = int(np.searchsorted(indices, offset, side="left"))
            right = int(np.searchsorted(indices, end, side="left"))
            if right > left:
                local_indices = torch.from_numpy(indices[left:right] - offset).to(parameter.device)
                local_values = torch.from_numpy(values[left:right]).to(
                    device=parameter.device, dtype=parameter.dtype
                )
                parameter.view(-1).index_add_(0, local_indices, -local_values)
            offset = end

def compute_layer_norms(named_params: Mapping, n_layers: int = 28) -> list[float]:
    """Compute true L2 gradient norms for parameters under ``blocks.<n>``."""
    sums = [0.0] * n_layers
    pattern = re.compile(r"(?:^|\.)blocks\.(\d+)\.")
    for name, value in named_params.items():
        gradient, _ = _gradient_and_size(value)
        match = pattern.search(name)
        if gradient is None or match is None:
            continue
        layer_index = int(match.group(1))
        if 0 <= layer_index < n_layers:
            safe = torch.nan_to_num(gradient.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            sums[layer_index] += float(torch.sum(safe * safe).cpu())
    return [value**0.5 for value in sums]
