from typing import Optional


def assign_training_slice(
    worker_index: int,
    num_workers: int,
    total_steps: int,
    data_length: int,
    seq_len: int,
) -> dict:
    if num_workers == 0:
        return {"worker_index": 0, "start_step": 0, "end_step": 0, "start_token": 0, "end_token": 0, "total_tokens": 0}
    steps_per_worker = total_steps // num_workers
    remainder = total_steps % num_workers
    start_step = worker_index * steps_per_worker + min(worker_index, remainder)
    end_step = start_step + steps_per_worker + (1 if worker_index < remainder else 0)
    tokens_per_step = seq_len
    start_token = start_step * tokens_per_step
    end_token = end_step * tokens_per_step
    return {
        "worker_index": worker_index,
        "start_step": start_step,
        "end_step": end_step,
        "start_token": start_token,
        "end_token": end_token,
        "total_tokens": end_token - start_token,
    }


def distribute_data_across_workers(
    num_workers: int,
    total_steps: int,
    seq_len: int = 1024,
) -> list[dict]:
    return [
        assign_training_slice(i, num_workers, total_steps, 0, seq_len)
        for i in range(num_workers)
    ]
