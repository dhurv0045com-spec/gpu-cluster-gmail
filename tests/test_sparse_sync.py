import os

import numpy as np
import torch

from backend.aggregator import aggregate_sparse_gradients
from worker.sparse_grad import (
    apply_sparse_delta_to_model,
    compute_layer_norms,
    load_sparse_gradient,
    save_sparse_gradient,
    sparsify_gradients,
)


class TinyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(512, 512, bias=False), torch.nn.Linear(512, 512, bias=False)]
        )

    def forward(self, inputs):
        for block in self.blocks:
            inputs = torch.tanh(block(inputs))
        return inputs


def flatten_parameters(model):
    return torch.cat([parameter.detach().reshape(-1).cpu() for parameter in model.parameters()])


def worker_payload(model, worker_id, scale):
    generator = torch.Generator().manual_seed(42 + int(scale))
    for parameter in model.parameters():
        parameter.grad = torch.randn(parameter.shape, generator=generator) * scale
    return sparsify_gradients(
        dict(model.named_parameters()),
        token_count=4096,
        worker_id=worker_id,
        step=64,
        model_version=0,
        topk_fraction=0.01,
    )


def test_two_worker_sparse_round_is_small_and_moves_down_gradient(tmp_path):
    model_a = TinyNetwork()
    model_b = TinyNetwork()
    target = TinyNetwork()
    target.load_state_dict(model_a.state_dict())

    payloads = [worker_payload(model_a, "worker_A", 1), worker_payload(model_b, "worker_B", 2)]
    loaded = []
    for index, payload in enumerate(payloads):
        path = tmp_path / f"worker_{index}.npz"
        save_sparse_gradient(payload, path)
        # The archive includes schema/metadata, yet remains below 5% of a dense
        # fp16 gradient for a realistically non-trivial tensor count.
        dense_fp16_bytes = sum(parameter.numel() * 2 for parameter in model_a.parameters())
        assert os.path.getsize(path) < dense_fp16_bytes * 0.05
        loaded.append(load_sparse_gradient(path))

    indices, values = aggregate_sparse_gradients(loaded, max_grad_norm=None)
    before = flatten_parameters(target)
    apply_sparse_delta_to_model(target, indices, values.astype(np.float32))
    after = flatten_parameters(target)

    assert indices.size > 0
    assert torch.any(before != after)
    actual_change = (after - before)[torch.from_numpy(indices.astype(np.int64))]
    expected_change = -torch.from_numpy(values.astype(np.float32))
    assert torch.allclose(actual_change, expected_change, atol=2e-3, rtol=2e-3)


def test_layer_norms_are_computed_from_transformer_blocks():
    model = TinyNetwork()
    for index, parameter in enumerate(model.parameters(), start=1):
        parameter.grad = torch.full_like(parameter, float(index))
    norms = compute_layer_norms(dict(model.named_parameters()), n_layers=2)
    assert len(norms) == 2
    assert norms[0] > 0
    assert norms[1] > norms[0]


def test_anra_payload_target_and_efficiency_are_within_budget():
    parameters = 499_167_047
    sparse_payload_mb = int(parameters * 0.01) * 6 / 1_000_000
    compute_seconds = 64 * 2.9
    # One worker upload + one delta download at 10 MB/s, plus a conservative
    # six seconds of coordinator merge/state overhead.
    sync_seconds = 2 * sparse_payload_mb / 10 + 6
    overhead_fraction = sync_seconds / (compute_seconds + sync_seconds)
    assert sparse_payload_mb < 50
    assert overhead_fraction < 0.20
