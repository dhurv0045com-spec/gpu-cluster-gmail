import json
import time
from pathlib import Path


def wait_for_aggregation(step: int, cluster_drive_folder: str, worker_id: str,
                         timeout_seconds: int = 300, poll_interval: float = 2.0,
                         max_backoff: float = 30.0) -> str:
    start_time = time.time()
    current_interval = poll_interval
    state_path = Path(cluster_drive_folder) / "coordinator_state.json"
    initial_version = None

    state = _read_state_safe(state_path)
    if state:
        initial_version = state.get("master_weights_version", 0)
        phase = state.get("phase", "")
        if phase == "aggregating":
            pass

    waited = 0
    while (time.time() - start_time) < timeout_seconds:
        if _should_stop():
            return None

        state = _read_state_safe(state_path)
        if state:
            current_version = state.get("master_weights_version", 0)
            if current_version > initial_version:
                weights_path = Path(cluster_drive_folder) / f"master_weights_v{current_version}.pt"
                if weights_path.exists():
                    return str(weights_path)

        time.sleep(current_interval)
        waited += current_interval
        current_interval = min(current_interval * 1.5, max_backoff)

        if waited >= 60 and int(waited) % 60 == 0:
            print(f"Still waiting for aggregation... ({int(waited)}s)")

    print(f"Timeout after {timeout_seconds}s waiting for aggregation at step {step}")
    fallback = get_latest_weights(cluster_drive_folder)
    if fallback:
        print(f"Using fallback weights: {fallback}")
        return fallback
    return None


def wait_for_sync_update(current_version: int, step: int, cluster_drive_folder: str,
                         worker_id: str, timeout_seconds: int = 300,
                         poll_interval: float = 2.0, max_backoff: float = 30.0) -> dict:
    """Wait on the mounted state file and return the complete sync manifest."""
    del worker_id  # Reserved for per-worker acknowledgements in future protocol versions.
    start_time = time.time()
    current_interval = poll_interval
    state_path = Path(cluster_drive_folder) / "coordinator_state.json"
    next_progress_log = 60

    while (time.time() - start_time) < timeout_seconds:
        if _should_stop():
            return {}
        state = _read_state_safe(state_path)
        if state.get("master_weights_version", 0) > current_version:
            return state

        time.sleep(current_interval)
        current_interval = min(current_interval * 1.5, max_backoff)
        elapsed = time.time() - start_time
        if elapsed >= next_progress_log:
            print(f"Still waiting for sparse aggregation at step {step}... ({int(elapsed)}s)")
            next_progress_log += 60

    print(f"Timeout after {timeout_seconds}s waiting for sparse aggregation at step {step}")
    state = _read_state_safe(state_path)
    return state if state.get("master_weights_version", 0) > current_version else {}


def get_latest_weights(cluster_drive_folder: str) -> str:
    weights_dir = Path(cluster_drive_folder)
    pt_files = list(weights_dir.glob("master_weights_v*.pt"))
    if not pt_files:
        return None
    pt_files.sort(key=lambda p: int(p.stem.split("_v")[-1]))
    return str(pt_files[-1])


def _should_stop():
    try:
        from drive_worker import should_stop
        return should_stop()
    except ImportError:
        return False


def _read_state_safe(state_path: Path) -> dict:
    try:
        if state_path.exists():
            with open(state_path) as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError, PermissionError):
        pass
    return {}
