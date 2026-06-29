import os
import sys
import time
import json
import torch
import torch.nn.functional as F
import requests
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drive_worker import (
    load_model_from_checkpoint, save_gradients_to_drive,
    get_latest_master_weights, get_gpu_memory_mb,
    read_coordinator_state, should_stop,
)
from wait_for_aggregation import wait_for_aggregation


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
        except (json.JSONDecodeError, IOError):
            pass
        return 0

    def _save_position(self, offset):
        try:
            with open(self.position_file, "w") as f:
                json.dump({"byte_offset": offset, "updated_at": time.time()}, f)
        except IOError:
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
                with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
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
):
    sys.path.insert(0, anra_repo_path)

    resp = requests.post(f"{coordinator_url}/api/workers/register", json={
        "worker_id": worker_id,
        "account_email": account_email,
        "drive_folder_id": cluster_drive_folder,
    }, timeout=30)
    resp.raise_for_status()
    assignment = resp.json()
    print(f"Registered as slot {assignment['assigned_slot']}")

    master_weights = assignment.get("master_weights_path") or checkpoint_path
    if not os.path.exists(master_weights):
        master_weights = get_latest_master_weights(cluster_drive_folder)
        if not master_weights:
            master_weights = checkpoint_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model = load_model_from_checkpoint(master_weights, model_class, device, model_kwargs)

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
    model_version = 0
    consecutive_errors = 0

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
                new_weights = get_latest_master_weights(cluster_drive_folder)
                if new_weights:
                    print(f"Reloading weights from {new_weights}")
                    model = load_model_from_checkpoint(new_weights, model_class, device, model_kwargs)
                    model_version += 1
                continue
            elif command == "pause":
                print("Paused by coordinator...")
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
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()

            grad_path = save_gradients_to_drive(
                model, step, worker_id, cluster_drive_folder,
                model_version, input_ids.numel(), last_loss,
                use_fp16=use_fp16_gradients,
            )

            try:
                requests.post(f"{coordinator_url}/api/workers/{worker_id}/gradient_ready", json={
                    "step": step,
                    "grad_file_path": grad_path,
                    "minibatch_tokens": input_ids.numel(),
                }, timeout=30)
            except requests.RequestException as e:
                print(f"Warning: Failed to signal coordinator: {e}")

            new_weights_path = wait_for_aggregation(
                step, cluster_drive_folder, worker_id, timeout_seconds=300
            )

            if new_weights_path:
                model = load_model_from_checkpoint(new_weights_path, model_class, device, model_kwargs)
                model_version += 1

            last_loss = loss.item()
            step += 1
            consecutive_errors = 0
            print(f"Step {step} | Loss: {last_loss:.4f} | GPU: {get_gpu_memory_mb():.0f}MB | v{model_version}")

        except Exception as e:
            consecutive_errors += 1
            print(f"Error at step {step}: {e}")
            if consecutive_errors > 5:
                print("Too many consecutive errors, stopping")
                break
            time.sleep(5)

    print(f"Worker loop ended. Completed {step} steps.")
