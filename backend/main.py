import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import auth as auth_module
from .aggregator import (
    aggregate_sparse_gradients,
    apply_sparse_delta_to_checkpoint,
    load_sparse_gradient,
)
from .database import ClusterState, init_db
from .drive_sync import DriveSync
from .worker_registry import (
    get_active_workers,
    get_all_workers,
    heartbeat,
    reap_stale_workers,
    register_worker,
    should_aggregate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("anra-coordinator")

drive_sync_instance = None
log_stream_queue: asyncio.Queue = None
aggregation_in_progress = False
executor = ThreadPoolExecutor(max_workers=2)
shutdown_event = asyncio.Event()
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173").rstrip("/")
CHECKPOINT_EVERY_N_SYNCS = int(os.environ.get("CHECKPOINT_EVERY_N_SYNCS", "10"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    global log_stream_queue
    log_stream_queue = asyncio.Queue(maxsize=1000)
    logger.info("Coordinator started")
    yield
    shutdown_event.set()
    executor.shutdown(wait=False)
    logger.info("Coordinator shut down")


app = FastAPI(
    title="AN-RA Cluster Coordinator",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "path": request.url.path},
    )


def get_drive_sync() -> DriveSync:
    if drive_sync_instance is None:
        raise HTTPException(status_code=503, detail="Drive not initialized. Call /api/cluster/init first.")
    return drive_sync_instance


async def emit_log(message: str, level: str = "INFO"):
    if log_stream_queue:
        try:
            await asyncio.wait_for(
                log_stream_queue.put(f"[{level}] [{datetime.now(UTC).isoformat()}] {message}"),
                timeout=1.0,
            )
        except (TimeoutError, asyncio.QueueFull):
            pass


async def run_in_thread(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: fn(*args, **kwargs))


class ClusterInitRequest(BaseModel):
    coordinator_drive_folder_id: str = Field(min_length=1)
    master_checkpoint_filename: str = Field(min_length=1)
    total_target_steps: int = Field(gt=0, le=10_000_000)


class WorkerRegisterRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    account_email: str = Field(min_length=3)
    drive_folder_id: str = Field(min_length=1)


class HeartbeatRequest(BaseModel):
    current_step: int = Field(ge=0)
    loss: float | None = None
    tokens_processed: int | None = Field(default=None, ge=0)
    gpu_memory_mb: float | None = Field(default=None, ge=0)


class GradientReadyRequest(BaseModel):
    step: int = Field(ge=0)
    grad_file_path: str = Field(min_length=1)
    minibatch_tokens: int = Field(ge=1)
    model_version: int = Field(default=0, ge=0)
    layer_norms: list[float] = Field(default_factory=list, max_length=256)


class AggregateRequest(BaseModel):
    step: int = Field(ge=0)


@app.get("/api/auth/login")
async def auth_login():
    try:
        authorization_url, oauth_state = auth_module.get_authorization_url()
        auth_module.store_oauth_state(oauth_state)
        return {"authorization_url": authorization_url, "state": oauth_state}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Google OAuth client secrets are not configured.") from exc


@app.get("/api/auth/callback")
async def auth_callback(code: str, state: str):
    if not auth_module.consume_oauth_state(state):
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state. Start Drive connection again.")
    try:
        credentials = auth_module.exchange_code_for_token(code)
        auth_module.store_credentials(auth_module.credentials_to_dict(credentials))
    except Exception as exc:
        logger.warning("OAuth callback failed: %s", exc)
        raise HTTPException(status_code=400, detail="Google Drive authorization failed. Please try again.") from exc
    return RedirectResponse(url=f"{FRONTEND_URL}/setup?auth=success")


@app.get("/api/auth/status")
async def auth_status():
    try:
        credentials = auth_module.load_credentials()
    except Exception as exc:
        logger.warning("Stored Drive credentials could not be refreshed: %s", exc)
        return {"authenticated": False}
    return {"authenticated": credentials is not None and credentials.valid}


@app.post("/api/cluster/init", status_code=201)
async def cluster_init(req: ClusterInitRequest):
    global drive_sync_instance
    from sqlmodel import Session

    from .database import engine

    try:
        credentials = auth_module.load_credentials()
    except Exception as exc:
        logger.warning("Drive credential refresh failed: %s", exc)
        credentials = None
    if credentials is None or not credentials.valid:
        raise HTTPException(status_code=401, detail="Drive not authenticated. Visit /api/auth/login first.")

    candidate_drive_sync = DriveSync(credentials, req.coordinator_drive_folder_id)

    if not await run_in_thread(candidate_drive_sync.file_exists, req.master_checkpoint_filename):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Master checkpoint '{req.master_checkpoint_filename}' was not found in the cluster folder. "
                "Upload it before initialization so every worker has the same recovery base."
            ),
        )

    state_data = {
        "global_step": 0,
        "phase": "waiting_for_workers",
        "expected_workers": [],
        "submitted_this_step": [],
        "submission_times": {},
        "gradient_submissions": {},
        "master_weights_version": 0,
        "checkpoint_version": 0,
        "master_weights_path": req.master_checkpoint_filename,
        "delta_path": "",
        "delta_history": [],
        "latest_layer_norms": [],
        "current_lr": 3e-4,
        "lock_holder": None,
        "lock_time": None,
        "total_target_steps": req.total_target_steps,
    }
    # Do the authenticated write before committing SQL state. A bad folder ID
    # or revoked grant must not leave a half-initialized cluster behind.
    await run_in_thread(candidate_drive_sync.write_coordinator_state, state_data)
    drive_sync_instance = candidate_drive_sync

    with Session(engine) as session:
        state = session.get(ClusterState, 1)
        if state is None:
            state = ClusterState(id=1)
        state.coordinator_drive_folder_id = req.coordinator_drive_folder_id
        state.master_checkpoint_filename = req.master_checkpoint_filename
        state.total_target_steps = req.total_target_steps
        state.global_step = 0
        state.phase = "waiting_for_workers"
        state.master_weights_version = 0
        state.master_weights_path = req.master_checkpoint_filename
        session.add(state)
        session.commit()
    await emit_log(f"Cluster initialized. Target: {req.total_target_steps} steps.")
    return {"cluster_id": "anra-cluster-1", "status": "initialized"}


@app.post("/api/workers/register", status_code=201)
async def worker_register(req: WorkerRegisterRequest, request: Request):
    ip = request.client.host if request.client else None
    result = register_worker(req.worker_id, req.account_email, req.drive_folder_id, ip)
    drive_sync = get_drive_sync()
    state = drive_sync.read_coordinator_state()
    if req.worker_id not in state.get("expected_workers", []):
        state.setdefault("expected_workers", []).append(req.worker_id)
        if state.get("phase") == "waiting_for_workers":
            state["phase"] = "training"
        drive_sync.write_coordinator_state(state)
    await emit_log(f"Worker registered: {req.worker_id} (slot {result['assigned_slot']})")
    return {
        "assigned_slot": result["assigned_slot"],
        "master_weights_path": state.get("master_weights_path", ""),
        "master_weights_version": state.get("master_weights_version", 0),
        "checkpoint_version": state.get("checkpoint_version", 0),
    }


@app.get("/api/workers")
async def list_workers():
    reap_stale_workers()
    return get_all_workers()


@app.post("/api/workers/{worker_id}/heartbeat")
async def worker_heartbeat(worker_id: str, req: HeartbeatRequest):
    command = heartbeat(worker_id, req.current_step, req.loss, req.tokens_processed, req.gpu_memory_mb)
    return {"command": command}


@app.post("/api/workers/{worker_id}/gradient_ready")
async def gradient_ready(worker_id: str, req: GradientReadyRequest):
    global aggregation_in_progress
    drive_sync = get_drive_sync()
    state = drive_sync.read_coordinator_state()
    current_version = int(state.get("master_weights_version", 0))
    if req.model_version != current_version:
        raise HTTPException(
            status_code=409,
            detail=f"Worker model v{req.model_version} is stale; coordinator is at v{current_version}.",
        )
    submitted = set(state.get("submitted_this_step", []))
    submission_times = state.get("submission_times", {})
    submissions = state.get("gradient_submissions", {})

    if worker_id not in submitted:
        submitted.add(worker_id)
        state["submitted_this_step"] = list(submitted)
        if worker_id not in submission_times:
            submission_times[worker_id] = time.time()
        state["submission_times"] = submission_times
        submissions[worker_id] = {
            "step": req.step,
            "grad_file_path": os.path.basename(req.grad_file_path),
            "minibatch_tokens": req.minibatch_tokens,
            "model_version": req.model_version,
            "layer_norms": req.layer_norms,
        }
        state["gradient_submissions"] = submissions
        drive_sync.write_coordinator_state(state)
        await emit_log(f"Gradient received from {worker_id} for step {req.step}")

    earliest = min(submission_times.values()) if submission_times else None
    if should_aggregate(req.step, submitted, earliest) and not aggregation_in_progress:
        asyncio.create_task(run_aggregation(req.step))
        await emit_log(f"Triggering aggregation for step {req.step}")
        return {"acknowledged": True, "aggregation_pending": True}

    return {"acknowledged": True, "aggregation_pending": False}


def _update_sql_state(**updates):
    from sqlmodel import Session

    from .database import engine

    with Session(engine) as session:
        cluster_state = session.get(ClusterState, 1)
        if cluster_state is None:
            cluster_state = ClusterState(id=1)
        for key, value in updates.items():
            setattr(cluster_state, key, value)
        session.add(cluster_state)
        session.commit()


def _weighted_layer_norms(submissions: dict) -> list[float]:
    usable = [item for item in submissions.values() if item.get("layer_norms")]
    if not usable:
        return []
    count = max(len(item["layer_norms"]) for item in usable)
    total_tokens = sum(max(1, int(item.get("minibatch_tokens", 1))) for item in usable)
    result = np.zeros(count, dtype=np.float64)
    for item in usable:
        weight = max(1, int(item.get("minibatch_tokens", 1))) / total_tokens
        norms = np.asarray(item["layer_norms"], dtype=np.float64)
        result[: len(norms)] += np.nan_to_num(norms, nan=0.0, posinf=0.0, neginf=0.0) * weight
    return result.tolist()


def _materialize_checkpoint(drive_sync: DriveSync, state: dict, global_step: int) -> str:
    checkpoint_version = int(state.get("checkpoint_version", 0))
    checkpoint_name = state.get("master_weights_path", "")
    if not checkpoint_name:
        raise FileNotFoundError("No base checkpoint is configured")
    checkpoint_path = drive_sync.download_file(checkpoint_name)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in loaded and "state_dict" not in loaded:
        loaded = {"model_state_dict": loaded}

    for entry in sorted(state.get("delta_history", []), key=lambda item: int(item["version"])):
        if int(entry["version"]) <= checkpoint_version:
            continue
        delta_path = drive_sync.download_file(entry["path"])
        delta = load_sparse_gradient(delta_path)
        apply_sparse_delta_to_checkpoint(
            loaded,
            delta["indices"],
            delta["values"] * delta.get("learning_rate", 1.0),
            delta["parameter_names"],
            delta["parameter_sizes"],
        )

    version = int(state["master_weights_version"])
    loaded["global_step"] = global_step
    loaded["master_weights_version"] = version
    destination = os.path.join(tempfile.gettempdir(), f"master_weights_v{version}.pt")
    torch.save(loaded, destination)
    drive_sync.write_master_weights(destination, version)
    return os.path.basename(destination)


async def run_aggregation(step: int):
    global aggregation_in_progress
    if aggregation_in_progress:
        await emit_log("Aggregation already in progress, skipping", "WARN")
        return
    aggregation_in_progress = True

    drive_sync = get_drive_sync()
    lock_holder = f"coordinator-{uuid.uuid4().hex[:12]}"
    lock_acquired = False
    state = {}
    try:
        lock_acquired = await run_in_thread(drive_sync.acquire_lock, lock_holder)
        if not lock_acquired:
            await emit_log("Another coordinator owns the Drive aggregation lock", "WARN")
            return

        state = drive_sync.read_coordinator_state()
        state["phase"] = "aggregating"
        drive_sync.write_coordinator_state(state)
        _update_sql_state(phase="aggregating")
        await emit_log(f"Aggregation started for step {step}")

        submitted = state.get("submitted_this_step", [])
        submissions = state.get("gradient_submissions", {})
        sparse_gradients = []
        accepted_submissions = {}
        for worker_id in submitted:
            submission = submissions.get(worker_id, {"step": step})
            try:
                path = await run_in_thread(
                    drive_sync.read_gradient_file, worker_id, int(submission.get("step", step))
                )
                sparse_gradient = await run_in_thread(load_sparse_gradient, path)
                sparse_gradients.append(sparse_gradient)
                accepted_submissions[worker_id] = submission
            except (FileNotFoundError, ValueError) as exc:
                await emit_log(f"Invalid gradient for {worker_id}, skipping: {exc}", "WARN")

        if not sparse_gradients:
            await emit_log("No sparse gradient files found, aborting aggregation", "WARN")
            state.update(
                phase="training", submitted_this_step=[], submission_times={}, gradient_submissions={}
            )
            drive_sync.write_coordinator_state(state)
            _update_sql_state(phase="training")
            return

        model_versions = {gradient["model_version"] for gradient in sparse_gradients}
        expected_version = int(state.get("master_weights_version", 0))
        if model_versions != {expected_version}:
            raise ValueError(
                f"Gradient model versions {sorted(model_versions)} do not match coordinator v{expected_version}"
            )

        indices, gradient_values = await run_in_thread(
            aggregate_sparse_gradients, sparse_gradients, 1.0
        )
        new_version = expected_version + 1
        learning_rate = float(state.get("current_lr", 3e-4))
        aggregated_step = max(gradient["step"] for gradient in sparse_gradients)
        schema_source = sparse_gradients[0]
        delta_path = os.path.join(tempfile.gettempdir(), f"delta_v{new_version:06d}.npz")
        np.savez_compressed(
            delta_path,
            indices=indices,
            values=gradient_values,
            total_params=np.array([schema_source["total_params"]], dtype=np.int64),
            token_count=np.array([sum(item["token_count"] for item in sparse_gradients)], dtype=np.int64),
            worker_id=np.array(["coordinator"]),
            step=np.array([aggregated_step], dtype=np.int64),
            model_version=np.array([new_version], dtype=np.int64),
            parameter_names=np.asarray(schema_source["parameter_names"]),
            parameter_sizes=np.asarray(schema_source["parameter_sizes"], dtype=np.int64),
            learning_rate=np.array([learning_rate], dtype=np.float32),
        )
        delta_name = await run_in_thread(drive_sync.write_sparse_delta, delta_path, new_version)

        delta_history = list(state.get("delta_history", []))
        delta_history.append({"version": new_version, "path": delta_name})
        state.update(
            global_step=aggregated_step,
            master_weights_version=new_version,
            delta_path=delta_name,
            delta_history=delta_history,
            latest_layer_norms=_weighted_layer_norms(accepted_submissions),
            submitted_this_step=[],
            submission_times={},
            gradient_submissions={},
        )
        # Publish the small delta first. Workers can see and apply it while the
        # occasional crash-recovery snapshot is being materialized.
        drive_sync.write_coordinator_state(state)

        if CHECKPOINT_EVERY_N_SYNCS > 0 and new_version % CHECKPOINT_EVERY_N_SYNCS == 0:
            try:
                checkpoint_name = await run_in_thread(
                    _materialize_checkpoint, drive_sync, state, aggregated_step
                )
                state["master_weights_path"] = checkpoint_name
                state["checkpoint_version"] = new_version
                # Keep the newest delta so a worker on v(N-1) still takes the
                # 30 MB fast path instead of reloading the fresh 2 GB snapshot.
                state["delta_history"] = [delta_history[-1]]
                await emit_log(f"Crash-recovery checkpoint v{new_version} published")
            except Exception as exc:
                await emit_log(f"Checkpoint snapshot deferred: {exc}", "WARN")
                logger.exception("Periodic checkpoint materialization failed")

        target_steps = int(state.get("total_target_steps", 0))
        final_phase = "completed" if target_steps > 0 and aggregated_step >= target_steps else "training"
        state["phase"] = final_phase
        drive_sync.write_coordinator_state(state)
        await run_in_thread(drive_sync.cleanup_old_gradients, 3)
        _update_sql_state(
            global_step=aggregated_step,
            master_weights_version=new_version,
            master_weights_path=state.get("master_weights_path", ""),
            phase=final_phase,
        )
        await emit_log(f"Aggregation complete. Step {aggregated_step}, sparse delta v{new_version}")

    except Exception as exc:
        await emit_log(f"Aggregation error: {exc}", "ERROR")
        logger.exception("Aggregation failed")
        if state:
            state.update(
                phase="training", submitted_this_step=[], submission_times={}, gradient_submissions={}
            )
            drive_sync.write_coordinator_state(state)
        _update_sql_state(phase="training")
    finally:
        if lock_acquired:
            try:
                await run_in_thread(drive_sync.release_lock, lock_holder)
            except Exception:
                logger.exception("Failed to release Drive aggregation lock")
        aggregation_in_progress = False


@app.post("/api/training/pause")
async def pause_training():
    from sqlmodel import Session

    from .database import engine
    with Session(engine) as session:
        cs = session.get(ClusterState, 1)
        if cs:
            cs.phase = "paused"
            session.add(cs)
            session.commit()
    drive_sync = get_drive_sync()
    state = drive_sync.read_coordinator_state()
    state["phase"] = "paused"
    drive_sync.write_coordinator_state(state)
    await emit_log("Training paused by user", "INFO")
    return {"status": "paused"}


@app.post("/api/training/resume")
async def resume_training():
    from sqlmodel import Session

    from .database import engine
    with Session(engine) as session:
        cs = session.get(ClusterState, 1)
        if cs:
            cs.phase = "training"
            session.add(cs)
            session.commit()
    drive_sync = get_drive_sync()
    state = drive_sync.read_coordinator_state()
    state["phase"] = "training"
    drive_sync.write_coordinator_state(state)
    await emit_log("Training resumed by user", "INFO")
    return {"status": "resumed"}


@app.post("/api/training/aggregate")
async def manual_aggregate(req: AggregateRequest):
    if aggregation_in_progress:
        raise HTTPException(status_code=409, detail="Aggregation already in progress")
    asyncio.create_task(run_aggregation(req.step))
    return {"status": "aggregation_started", "step": req.step}


@app.get("/api/training/status")
async def training_status():
    drive_sync = get_drive_sync()
    state = drive_sync.read_coordinator_state()
    reap_stale_workers()
    workers = get_all_workers()
    active = get_active_workers()

    loss_history = state.get("total_loss_history", "")
    if isinstance(loss_history, str) and loss_history:
        try:
            loss_list = json.loads(loss_history)
        except (json.JSONDecodeError, TypeError):
            loss_list = []
    elif isinstance(loss_history, list):
        loss_list = loss_history
    else:
        loss_list = []

    total_tokens = sum(
        w.get("tokens_processed", 0) for w in workers if w.get("status") == "active"
    )
    gs = state.get("global_step", 0) or 1

    return {
        "global_step": state.get("global_step", 0),
        "total_target_steps": state.get("total_target_steps", 0),
        "total_loss_history": loss_list,
        "active_workers": len(active),
        "total_workers": len(workers),
        "estimated_eta": state.get("estimated_eta", 0),
        "tokens_per_second_total": state.get("tokens_per_second_total", 0) or total_tokens / gs,
        "phase": state.get("phase", "idle"),
        "master_weights_version": state.get("master_weights_version", 0),
        "current_lr": state.get("current_lr", 3e-4),
        "layer_norms": state.get("latest_layer_norms", []),
        "checkpoint_version": state.get("checkpoint_version", 0),
        "delta_path": state.get("delta_path", ""),
        "aggregation_in_progress": aggregation_in_progress,
    }


@app.get("/api/drive/files")
async def drive_files():
    drive_sync = get_drive_sync()
    try:
        files = await run_in_thread(drive_sync.list_all_files)
        return [
            {
                "id": f["id"],
                "name": f["name"],
                "mimeType": f.get("mimeType"),
                "size": f.get("size"),
                "createdTime": f.get("createdTime"),
            }
            for f in files
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs/stream")
async def log_stream(request: Request):
    async def event_generator():
        while not shutdown_event.is_set():
            try:
                message = await asyncio.wait_for(log_stream_queue.get(), timeout=30.0)
                yield f"data: {json.dumps({'message': message, 'timestamp': time.time()})}\n\n"
            except TimeoutError:
                yield f"data: {json.dumps({'type': 'keepalive', 'timestamp': time.time()})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "aggregation_in_progress": aggregation_in_progress,
        "drive_initialized": drive_sync_instance is not None,
    }
