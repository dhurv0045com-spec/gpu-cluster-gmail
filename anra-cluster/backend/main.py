import os
import json
import time
import asyncio
import torch
import tempfile
import signal
import logging
from datetime import datetime, timezone
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from database import init_db, ClusterState
from aggregator import aggregate_gradients
from worker_registry import (
    register_worker, heartbeat, get_all_workers, get_active_workers,
    should_aggregate, reap_stale_workers,
)
from drive_sync import DriveSync

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("anra-coordinator")

drive_sync_instance = None
log_stream_queue: asyncio.Queue = None
aggregation_in_progress = False
executor = ThreadPoolExecutor(max_workers=2)
shutdown_event = asyncio.Event()


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
                log_stream_queue.put(f"[{level}] [{datetime.now(timezone.utc).isoformat()}] {message}"),
                timeout=1.0,
            )
        except (asyncio.TimeoutError, asyncio.QueueFull):
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
    loss: Optional[float] = None
    tokens_processed: Optional[int] = Field(default=None, ge=0)
    gpu_memory_mb: Optional[float] = Field(default=None, ge=0)


class GradientReadyRequest(BaseModel):
    step: int = Field(ge=0)
    grad_file_path: str = Field(min_length=1)
    minibatch_tokens: int = Field(ge=1)


class AggregateRequest(BaseModel):
    step: int = Field(ge=0)


@app.post("/api/cluster/init", status_code=201)
async def cluster_init(req: ClusterInitRequest):
    global drive_sync_instance
    from sqlmodel import Session
    from database import engine
    from google.oauth2.credentials import Credentials

    with Session(engine) as session:
        state = session.get(ClusterState, 1)
        if state is None:
            state = ClusterState(id=1)
        state.coordinator_drive_folder_id = req.coordinator_drive_folder_id
        state.master_checkpoint_filename = req.master_checkpoint_filename
        state.total_target_steps = req.total_target_steps
        state.global_step = 0
        state.phase = "initializing"
        state.master_weights_version = 0
        state.master_weights_path = ""
        session.add(state)
        session.commit()
        folder_id = state.coordinator_drive_folder_id

    dummy_creds = Credentials(token="placeholder")
    drive_sync_instance = DriveSync(dummy_creds, folder_id)

    state_data = {
        "global_step": 0,
        "phase": "waiting_for_workers",
        "expected_workers": [],
        "submitted_this_step": [],
        "submission_times": {},
        "master_weights_version": 0,
        "master_weights_path": req.master_checkpoint_filename,
        "current_lr": 3e-4,
        "lock_holder": None,
        "lock_time": None,
        "total_target_steps": req.total_target_steps,
    }
    drive_sync_instance.write_coordinator_state(state_data)
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
    return {"assigned_slot": result["assigned_slot"], "master_weights_path": state.get("master_weights_path", "")}


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
    submitted = set(state.get("submitted_this_step", []))
    submission_times = state.get("submission_times", {})

    if worker_id not in submitted:
        submitted.add(worker_id)
        state["submitted_this_step"] = list(submitted)
        if worker_id not in submission_times:
            submission_times[worker_id] = time.time()
        state["submission_times"] = submission_times
        drive_sync.write_coordinator_state(state)
        await emit_log(f"Gradient received from {worker_id} for step {req.step}")

    earliest = min(submission_times.values()) if submission_times else None
    if should_aggregate(req.step, submitted, earliest) and not aggregation_in_progress:
        asyncio.create_task(run_aggregation(req.step))
        await emit_log(f"Triggering aggregation for step {req.step}")
        return {"acknowledged": True, "aggregation_pending": True}

    return {"acknowledged": True, "aggregation_pending": False}


async def run_aggregation(step: int):
    global aggregation_in_progress
    if aggregation_in_progress:
        await emit_log("Aggregation already in progress, skipping", "WARN")
        return
    aggregation_in_progress = True

    drive_sync = get_drive_sync()
    state = drive_sync.read_coordinator_state()
    state["phase"] = "aggregating"
    drive_sync.write_coordinator_state(state)
    await emit_log(f"Aggregation started for step {step}")

    try:
        submitted = state.get("submitted_this_step", [])
        grad_paths = []
        for worker_id in submitted:
            try:
                path = drive_sync.read_gradient_file(worker_id, step)
                grad_paths.append(path)
            except FileNotFoundError:
                await emit_log(f"Gradient file not found for {worker_id}, skipping", "WARN")

        if not grad_paths:
            await emit_log("No gradient files found, aborting aggregation", "WARN")
            state["phase"] = "training"
            state["submitted_this_step"] = []
            state["submission_times"] = {}
            drive_sync.write_coordinator_state(state)
            return

        model_versions = set()
        for path in grad_paths:
            data = await run_in_thread(
                lambda p=path: torch.load(p, map_location="cpu", weights_only=True)
            )
            if isinstance(data, dict):
                if "model_version" in data:
                    model_versions.add(data["model_version"])

        if len(model_versions) > 1:
            await emit_log(f"Model version mismatch: {model_versions}. Proceeding.", "WARN")

        averaged = await run_in_thread(
            aggregate_gradients, grad_paths, None, 1.0
        )
        new_version = state.get("master_weights_version", 0) + 1
        lr = state.get("current_lr", 3e-4)

        master_weights_name = state.get("master_weights_path", "")
        master_weights_local = None
        if master_weights_name:
            master_weights_local = os.path.join(tempfile.gettempdir(), master_weights_name)

        checkpoint = None
        if master_weights_local and os.path.exists(master_weights_local):
            checkpoint = await run_in_thread(
                lambda: torch.load(master_weights_local, map_location="cpu", weights_only=False)
            )
        else:
            alt = os.path.join(tempfile.gettempdir(), "checkpoint_base.pt")
            if os.path.exists(alt):
                checkpoint = await run_in_thread(
                    lambda: torch.load(alt, map_location="cpu", weights_only=False)
                )

        if checkpoint is None:
            model_state = {k: torch.zeros_like(v) for k, v in averaged.items()}
            optim_state = {}
            await emit_log("No master weights found, creating from scratch", "WARN")
        else:
            model_state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", {}))
            optim_state = checkpoint.get("optimizer_state_dict", {})

        for name, grad in averaged.items():
            if name in model_state:
                model_state[name].sub_(grad.float().to(model_state[name].dtype), alpha=lr)

        new_checkpoint_path = os.path.join(
            tempfile.gettempdir(), f"master_weights_v{new_version}.pt"
        )
        save_data = {
            "model_state_dict": model_state,
            "optimizer_state_dict": optim_state,
            "global_step": step,
        }
        await run_in_thread(
            lambda: torch.save(save_data, new_checkpoint_path)
        )

        await run_in_thread(
            lambda: drive_sync.write_master_weights(new_checkpoint_path, new_version)
        )
        drive_sync.cleanup_old_gradients(keep_last_n_steps=3)

        state["global_step"] = step
        state["master_weights_version"] = new_version
        state["master_weights_path"] = f"master_weights_v{new_version}.pt"
        state["phase"] = "training"
        state["submitted_this_step"] = []
        state["submission_times"] = {}
        drive_sync.write_coordinator_state(state)
        await emit_log(f"Aggregation complete. Step {step}, master weights v{new_version}")

        from sqlmodel import Session
        from database import engine
        with Session(engine) as session:
            cs = session.get(ClusterState, 1)
            if cs:
                cs.global_step = step
                cs.master_weights_version = new_version
                cs.phase = "training"
                session.add(cs)
                session.commit()

    except Exception as e:
        await emit_log(f"Aggregation error: {e}", "ERROR")
        logger.exception("Aggregation failed")
        state["phase"] = "training"
        state["submitted_this_step"] = []
        state["submission_times"] = {}
        drive_sync.write_coordinator_state(state)
    finally:
        aggregation_in_progress = False


@app.post("/api/training/pause")
async def pause_training():
    from sqlmodel import Session
    from database import engine
    with Session(engine) as session:
        cs = session.get(ClusterState, 1)
        if cs:
            cs.phase = "paused"
            session.add(cs)
            session.commit()
    state = drive_sync_instance.read_coordinator_state()
    state["phase"] = "paused"
    drive_sync_instance.write_coordinator_state(state)
    await emit_log("Training paused by user", "INFO")
    return {"status": "paused"}


@app.post("/api/training/resume")
async def resume_training():
    from sqlmodel import Session
    from database import engine
    with Session(engine) as session:
        cs = session.get(ClusterState, 1)
        if cs:
            cs.phase = "training"
            session.add(cs)
            session.commit()
    state = drive_sync_instance.read_coordinator_state()
    state["phase"] = "training"
    drive_sync_instance.write_coordinator_state(state)
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
            except asyncio.TimeoutError:
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
