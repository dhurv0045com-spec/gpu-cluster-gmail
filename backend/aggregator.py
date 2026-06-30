
import numpy as np
import torch


def load_sparse_gradient(path: str) -> dict:
    """Load and validate a worker's compressed sparse payload."""
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
        total_params = int(data["total_params"][0])
        if indices.ndim != 1 or indices.shape != values.shape:
            raise ValueError("Sparse indices and values must be matching one-dimensional arrays")
        if indices.size and (indices.min() < 0 or indices.max() >= total_params):
            raise ValueError("Sparse payload contains an out-of-range parameter index")
        return {
            "indices": indices,
            "values": values,
            "total_params": total_params,
            "token_count": int(data["token_count"][0]),
            "worker_id": str(data["worker_id"][0]),
            "step": int(data["step"][0]),
            "model_version": int(data["model_version"][0]),
            "parameter_names": data["parameter_names"].astype(str).tolist(),
            "parameter_sizes": data["parameter_sizes"].astype(np.int64).tolist(),
            "learning_rate": float(data["learning_rate"][0]) if "learning_rate" in data.files else 1.0,
        }


def aggregate_sparse_gradients(
    sparse_grads: list[dict], max_grad_norm: float | None = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Token-weight and merge sparse gradients without a 2 GB dense buffer."""
    if not sparse_grads:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float16)

    total_params = sparse_grads[0]["total_params"]
    if any(gradient["total_params"] != total_params for gradient in sparse_grads):
        raise ValueError("Workers submitted gradients for different model architectures")
    schema = (sparse_grads[0]["parameter_names"], sparse_grads[0]["parameter_sizes"])
    if any((gradient["parameter_names"], gradient["parameter_sizes"]) != schema for gradient in sparse_grads[1:]):
        raise ValueError("Workers submitted incompatible parameter schemas")

    token_counts = np.asarray([max(0, gradient["token_count"]) for gradient in sparse_grads], dtype=np.float64)
    total_tokens = float(token_counts.sum())
    if total_tokens <= 0:
        raise ValueError("Sparse gradients have no positive token counts")

    index_parts = []
    value_parts = []
    for gradient, token_count in zip(sparse_grads, token_counts):
        values = np.nan_to_num(
            gradient["values"].astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0
        )
        values = np.clip(values, -1e3, 1e3)
        index_parts.append(gradient["indices"].astype(np.int32, copy=False))
        value_parts.append(values * np.float32(token_count / total_tokens))

    indices = np.concatenate(index_parts)
    values = np.concatenate(value_parts)
    if indices.size == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float16)

    order = np.argsort(indices, kind="stable")
    indices = indices[order]
    values = values[order]
    unique_indices, starts = np.unique(indices, return_index=True)
    merged_values = np.add.reduceat(values, starts).astype(np.float32, copy=False)
    nonzero = merged_values != 0
    unique_indices = unique_indices[nonzero]
    merged_values = merged_values[nonzero]

    if max_grad_norm is not None and merged_values.size:
        norm = float(np.linalg.norm(merged_values.astype(np.float64)))
        if norm > max_grad_norm:
            merged_values *= np.float32(max_grad_norm / (norm + 1e-8))

    return unique_indices.astype(np.int32, copy=False), merged_values.astype(np.float16, copy=False)


def apply_sparse_delta_to_checkpoint(
    checkpoint: dict,
    indices: np.ndarray,
    values: np.ndarray,
    parameter_names: list[str],
    parameter_sizes: list[int],
) -> dict:
    """Fold one optimizer delta into a model checkpoint without model source."""
    model_state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.float32)
    if indices.size and np.any(indices[1:] < indices[:-1]):
        order = np.argsort(indices, kind="stable")
        indices, values = indices[order], values[order]

    if len(parameter_names) != len(parameter_sizes):
        raise ValueError("Parameter schema names and sizes have different lengths")
    for name, expected_size in zip(parameter_names, parameter_sizes):
        tensor = model_state.get(name)
        if tensor is None or tensor.numel() != expected_size:
            raise ValueError(f"Checkpoint parameter schema mismatch at {name}")
    total_parameters = sum(int(size) for size in parameter_sizes)
    if indices.size and (indices[0] < 0 or indices[-1] >= total_parameters):
        raise ValueError("Sparse delta contains an index outside the checkpoint schema")

    offset = 0
    with torch.no_grad():
        for name, expected_size in zip(parameter_names, parameter_sizes):
            end = offset + int(expected_size)
            tensor = model_state.get(name)
            left = int(np.searchsorted(indices, offset, side="left"))
            right = int(np.searchsorted(indices, end, side="left"))
            if right > left:
                local_indices = torch.from_numpy(indices[left:right] - offset).long()
                local_values = torch.from_numpy(values[left:right]).to(tensor.dtype)
                tensor.view(-1).index_add_(0, local_indices, -local_values)
            offset = end
    return checkpoint
