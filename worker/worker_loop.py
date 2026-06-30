import json
import os
import sys
import time
from pathlib import Path

import requests
import torch
from torch.nn import functional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drive_worker import (
    get_gpu_memory_mb,
    get_latest_master_weights,
    load_model_from_checkpoint,
    read_coordinator_state,
    save_gradients_to_drive,
    should_stop,
)
from sparse_grad import apply_sparse_delta_to_model, compute_layer_norms, load_sparse_gradient
from wait_for_aggregation import wait_for_sync_update

SYNC_EVERY_N_STEPS = 64
SPARSITY_TOPK_FRACTION = 0.01


def _build_adafactor(model, optimizer_factory=None, learning_rate: float = 1e-4):
    """Use An-Ra's supplied factory when injected, otherwise PyTorch Adafactor."""
    if optimizer_factory is not None:
        return optimizer_factory(model.parameters(), lr=learning_rate)
    adafactor = getattr(torch.optim, "Adafactor", None)
    if adafactor is None:
        raise RuntimeError(
            "This worker requires torch.optim.Adafactor (PyTorch 2.5+) or an optimizer_factory "
            "from An-Ra's training package."
        )
    return adafactor(model.parameters(), lr=learning_rate)


def _cluster_file(cluster_drive_folder: str, filename: str) -> str:
    return str(Path(cluster_drive_folder) / Path(filename).name)


def _apply_sync_manifest(model, manifest: dict, current_version: int, cluster_drive_folder: str,
                         model_class, device: str, model_kwargs: dict):
    """Apply a complete consecutive delta chain, reloading a snapshot if needed."""
    target_version = int(manifest.get("master_weights_version", current_version))
    if target_version <= current_version:
        return model, current_version, False

    history = {
        int(entry["version"]): entry
        for entry in manifest.get("delta_history", [])
        if "version" in entry and "path" in entry
    }
    needed = list(range(current_version + 1, target_version + 1))
    reloaded = False

    if not all(version in history for version in needed):
        checkpoint_version = int(manifest.get("checkpoint_version", 0))
        checkpoint_name = manifest.get("master_weights_path", "")
        checkpoint_path = _cluster_file(cluster_drive_folder, checkpoint_name) if checkpoint_name else ""
        if checkpoint_version <= current_version or not checkpoint_path or not os.path.exists(checkpoint_path):
            missing = [version for version in needed if version not in history]
            raise RuntimeError(f"Cannot recover missing delta versions {missing}; checkpoint is unavailable")
        model = load_model_from_checkpoint(checkpoint_path, model_class, device, model_kwargs)
        current_version = checkpoint_version
        needed = list(range(current_version + 1, target_version + 1))
        reloaded = True

    for version in needed:
        entry = history.get(version)
        if entry is None:
            raise RuntimeError(f"Sync manifest is missing delta v{version}")
        delta_path = _cluster_file(cluster_drive_folder, entry["path"])
        if not os.path.exists(delta_path):
            raise FileNotFoundError(f"Sparse delta not yet visible in Drive mount: {delta_path}")
        delta = load_sparse_gradient(delta_path)
        if delta["model_version"] != version:
            raise ValueError(f"Delta file version {delta['model_version']} does not match manifest v{version}")
        # Keep averaged gradients in fp16 on disk, but perform LR scaling in
        # fp32 here so small updates do not underflow during serialization.
        values = delta["values"] * delta.get("learning_rate", 1.0)
        apply_sparse_delta_to_model(model, delta["indices"], values)
        current_version = version
    return model, current_version, reloaded


def _do_sparse_sync_round(model, coordinator_url: str, worker_id: str, cluster_drive_folder: str,
                          step: int, model_version: int, accumulated_tokens: int,
                          sparsity_fraction: float, model_class, device: str, model_kwargs: dict):
    layer_norms = compute_layer_norms(dict(model.named_parameters()))
    gradient_path = save_gradients_to_drive(
        model,
        step,
        worker_id,
        cluster_drive_folder,
        model_version,
        accumulated_tokens,
        topk_fraction=sparsity_fraction,
    )
    try:
        response = requests.post(
            f"{coordinator_url}/api/workers/{worker_id}/gradient_ready",
            json={
                "step": step,
                "grad_file_path": gradient_path,
                "minibatch_tokens": accumulated_tokens,
                "model_version": model_version,
                "layer_norms": layer_norms,
            },
            timeout=30,
        )
        if response.status_code == 409:
            manifest = read_coordinator_state(cluster_drive_folder)
            if manifest.get("master_weights_version", 0) > model_version:
                model, model_version, reloaded = _apply_sync_manifest(
                    model,
                    manifest,
                    model_version,
                    cluster_drive_folder,
                    model_class,
                    device,
                    model_kwargs,
                )
                # The coordinator already advanced, so these accumulated
                # gradients are stale. Catching up is a successful round; the
                # caller will clear them before computing on the new version.
                return model, model_version, reloaded, True
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Sparse sync signal failed: {exc}")
        return model, model_version, False, False

    manifest = wait_for_sync_update(
        model_version, step, cluster_drive_folder, worker_id, timeout_seconds=300
    )
    if not manifest:
        return model, model_version, False, False
    try:
        model, model_version, reloaded = _apply_sync_manifest(
            model,
            manifest,
            model_version,
            cluster_drive_folder,
            model_class,
            device,
            model_kwargs,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Sparse sync update is not ready: {exc}")
        return model, model_version, False, False
    return model, model_version, reloaded, True


class PositionTrackingDataLoader:
    def __init__(self, training_data_path, tokenizer, seq_len=1024, batch_size=1,
                 position_file=None):
        self.path = training_data_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.position_file = position_file or (Path(training_data_path).parent / "data_position.json")
        self._position = self._load_position()

    def _load_position(self):
        try:
            if os.path.exists(self.position_file):
                with open(self.position_file) as f:
                    return json.load(f).get("byte_offset", 0)
        except (OSError, json.JSONDecodeError):
            pass
        return 0

    def _save_position(self, offset):
        try:
            with open(self.position_file, "w") as f:
                json.dump({"byte_offset": offset, "updated_at": time.time()}, f)
        except OSError:
            pass

    def __iter__(self):
        from torch.utils.data import DataLoader, IterableDataset

        class TextStreamDataset(IterableDataset):
            def __init__(self, path, tokenizer, seq_len, start_offset=0):
                self.path = path
                self.tokenizer = tokenizer
                self.seq_len = seq_len
                self.start_offset = start_offset

            def __iter__(self):
                with open(self.path, encoding="utf-8", errors="ignore") as f:
                    f.seek(self.start_offset)
                    buffer = ""
                    for line in f:
                        buffer += line
                        while len(buffer) >= self.seq_len * 2:
                            chunk = buffer[:self.seq_len]
                            buffer = buffer[self.seq_len:]
                            tokens = self.tokenizer.encode(chunk)
                            tokens = tokens[:self.seq_len]
                            if len(tokens) < self.seq_len:
                                tokens = tokens + [0] * (self.seq_len - len(tokens))
                            input_ids = torch.tensor(tokens, dtype=torch.long)
                            yield {"input_ids": input_ids, "labels": input_ids.clone()}

        dataset = TextStreamDataset(self.path, self.tokenizer, self.seq_len, self._position)
        self._loader = DataLoader(dataset, batch_size=self.batch_size)
        return iter(self._loader)

    def update_position(self, chars_consumed):
        self._position += chars_consumed
        self._save_position(self._position)


def run_worker_loop(
    coordinator_url: str,
    worker_id: str,
    account_email: str,
    anra_repo_path: str,
    checkpoint_path: str,
    training_data: str,
    cluster_drive_folder: str,
    model_class=None,
    tokenizer_class=None,
    model_kwargs: dict = None,
    use_fp16_gradients: bool = True,
    optimizer_factory=None,
    learning_rate: float = 1e-4,
    sync_every_n_steps: int = SYNC_EVERY_N_STEPS,
    sparsity_fraction: float = SPARSITY_TOPK_FRACTION,
):
    del use_fp16_gradients
    if sync_every_n_steps < 1:
        raise ValueError("sync_every_n_steps must be positive")
    sys.path.insert(0, anra_repo_path)

    resp = requests.post(f"{coordinator_url}/api/workers/register", json={
        "worker_id": worker_id,
        "account_email": account_email,
        "drive_folder_id": cluster_drive_folder,
    }, timeout=30)
    resp.raise_for_status()
    assignment = resp.json()
    print(f"Registered as slot {assignment['assigned_slot']}")

    master_weights_name = assignment.get("master_weights_path")
    mounted_master = _cluster_file(cluster_drive_folder, master_weights_name) if master_weights_name else ""
    master_weights = mounted_master if mounted_master and os.path.exists(mounted_master) else checkpoint_path
    if not os.path.exists(master_weights):
        master_weights = get_latest_master_weights(cluster_drive_folder)
        if not master_weights:
            master_weights = checkpoint_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model = load_model_from_checkpoint(master_weights, model_class, device, model_kwargs)

    coordinator_state = read_coordinator_state(cluster_drive_folder)
    model_version = int(coordinator_state.get("checkpoint_version", 0))
    if coordinator_state.get("master_weights_version", 0) > model_version:
        model, model_version, _ = _apply_sync_manifest(
            model, coordinator_state, model_version, cluster_drive_folder, model_class, device, model_kwargs
        )
    optimizer = _build_adafactor(model, optimizer_factory, learning_rate)
    optimizer.zero_grad(set_to_none=True)

    if tokenizer_class:
        tok_path = os.path.join(anra_repo_path, "tokenizer", "tokenizer_v3.json")
        tokenizer = tokenizer_class.from_pretrained(tok_path)
    else:
        from anra_brain import AnRaTokenizer
        tokenizer = AnRaTokenizer.from_pretrained(
            os.path.join(anra_repo_path, "tokenizer", "tokenizer_v3.json")
        )

    data_loader = PositionTrackingDataLoader(
        training_data, tokenizer, seq_len=1024, batch_size=1
    )
    data_iter = iter(data_loader)

    step = 0
    last_loss = 0.0
    consecutive_errors = 0
    accumulated_steps_since_sync = 0
    accumulated_tokens = 0

    while not should_stop():
        try:
            try:
                resp = requests.post(f"{coordinator_url}/api/workers/{worker_id}/heartbeat", json={
                    "current_step": step,
                    "loss": last_loss,
                    "tokens_processed": step * 1024,
                    "gpu_memory_mb": get_gpu_memory_mb(),
                }, timeout=15)
                command = resp.json().get("command", "continue")
            except requests.RequestException:
                command = "continue"

            if command == "stop":
                print("Coordinator says stop. Training complete.")
                break
            elif command == "reload_weights":
                manifest = read_coordinator_state(cluster_drive_folder)
                model, model_version, reloaded = _apply_sync_manifest(
                    model, manifest, model_version, cluster_drive_folder, model_class, device, model_kwargs
                )
                if reloaded:
                    optimizer = _build_adafactor(model, optimizer_factory, learning_rate)
                    optimizer.zero_grad(set_to_none=True)
                continue
            elif command == "pause":
                print("Paused by coordinator...")
                time.sleep(5)
                continue

            if accumulated_steps_since_sync >= sync_every_n_steps:
                model, model_version, reloaded, synced = _do_sparse_sync_round(
                    model,
                    coordinator_url,
                    worker_id,
                    cluster_drive_folder,
                    step,
                    model_version,
                    accumulated_tokens,
                    sparsity_fraction,
                    model_class,
                    device,
                    model_kwargs,
                )
                if synced:
                    if reloaded:
                        optimizer = _build_adafactor(model, optimizer_factory, learning_rate)
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_steps_since_sync = 0
                    accumulated_tokens = 0
                    consecutive_errors = 0
                    print(f"Sparse sync complete | step {step} | model v{model_version}")
                else:
                    print(f"Sync failed at step {step}; retaining accumulated gradients for retry")
                    time.sleep(5)
                continue

            try:
                batch = next(data_iter)
            except StopIteration:
                print("Data exhausted, restarting from beginning")
                data_iter = iter(data_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, _ = model(input_ids)
            loss = functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            (loss / sync_every_n_steps).backward()
            last_loss = loss.item()
            step += 1
            accumulated_steps_since_sync += 1
            accumulated_tokens += input_ids.numel()
            consecutive_errors = 0
            print(
                f"Step {step} | Loss: {last_loss:.4f} | GPU: {get_gpu_memory_mb():.0f}MB | "
                f"v{model_version} | accumulation {accumulated_steps_since_sync}/{sync_every_n_steps}"
            )

        except Exception as e:
            consecutive_errors += 1
            print(f"Error at step {step}: {e}")
            if consecutive_errors > 5:
                print("Too many consecutive errors, stopping")
                break
            time.sleep(5)

    print(f"Worker loop ended. Completed {step} steps.")
