import os
import json
import time
import signal
import torch
from pathlib import Path


_worker_should_stop = False


def _handle_signal(signum, frame):
    global _worker_should_stop
    _worker_should_stop = True
    print(f"\nSignal {signum} received, will stop after current step")


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def should_stop() -> bool:
    return _worker_should_stop


def get_latest_master_weights(cluster_drive_folder: str) -> str:
    weights_dir = Path(cluster_drive_folder)
    pt_files = list(weights_dir.glob("master_weights_v*.pt"))
    if not pt_files:
        return None
    pt_files.sort(key=lambda p: int(p.stem.split("_v")[-1]))
    return str(pt_files[-1])


def read_coordinator_state(cluster_drive_folder: str) -> dict:
    state_path = Path(cluster_drive_folder) / "coordinator_state.json"
    if not state_path.exists():
        return {}
    try:
        with open(state_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_gradients_to_drive(model, step: int, worker_id: str, cluster_drive_folder: str,
                            model_version: int, token_count: int, loss: float = None,
                            use_fp16: bool = True) -> str:
    grad_dict = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.cpu()
            if use_fp16:
                grad = grad.half()
            grad_dict[name] = grad.clone()

    save_data = {
        "step": step,
        "worker_id": worker_id,
        "model_version": model_version,
        "token_count": token_count,
        "loss": loss,
        "gradients": grad_dict,
        "use_fp16": use_fp16,
    }
    worker_dir = Path(cluster_drive_folder) / worker_id
    worker_dir.mkdir(parents=True, exist_ok=True)
    grad_path = worker_dir / f"grad_step_{step:06d}.pt"
    torch.save(save_data, str(grad_path))
    return str(grad_path)


def load_model_from_checkpoint(checkpoint_path: str, model_class, device: str = "cuda",
                                model_kwargs: dict = None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_kwargs = model_kwargs or {}
    model = model_class(**model_kwargs)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
    model = model.to(device)
    model.train()
    return model


def get_gpu_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        return torch.cuda.memory_allocated() / 1024 / 1024
    except RuntimeError:
        return 0.0
