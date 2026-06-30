import time

from sqlmodel import select

from .database import ClusterState, Worker, get_worker_session

WORKER_TIMEOUT = 120
GRADIENT_TIMEOUT = 300


def register_worker(worker_id: str, account_email: str, drive_folder_id: str, ip_address: str = None) -> dict:
    with get_worker_session() as session:
        existing = session.get(Worker, worker_id)
        if existing:
            existing.account_email = account_email
            existing.drive_folder_id = drive_folder_id
            existing.status = "active"
            existing.last_heartbeat = time.time()
            existing.ip_address = ip_address
            session.add(existing)
            session.commit()
            session.refresh(existing)
            return {"assigned_slot": existing.assigned_slot, "worker_id": existing.worker_id}

        count = len(session.exec(select(Worker)).all())
        assigned_slot = count + 1
        worker = Worker(
            worker_id=worker_id,
            account_email=account_email,
            drive_folder_id=drive_folder_id,
            assigned_slot=assigned_slot,
            status="active",
            last_heartbeat=time.time(),
            ip_address=ip_address,
        )
        session.add(worker)
        session.commit()
        session.refresh(worker)
        return {"assigned_slot": worker.assigned_slot, "worker_id": worker.worker_id}


def heartbeat(worker_id: str, current_step: int, loss: float = None,
              tokens_processed: int = None, gpu_memory_mb: float = None) -> str:
    with get_worker_session() as session:
        worker = session.get(Worker, worker_id)
        if not worker:
            return "stop"
        worker.last_heartbeat = time.time()
        worker.current_step = current_step
        if loss is not None:
            worker.loss = loss
        if tokens_processed is not None:
            worker.tokens_processed = tokens_processed
        if gpu_memory_mb is not None:
            worker.gpu_memory_mb = gpu_memory_mb
        worker.status = "active"
        session.add(worker)
        session.commit()

        state = session.get(ClusterState, 1)
        if not state:
            return "continue"
        if state.total_target_steps > 0 and state.global_step >= state.total_target_steps:
            return "stop"
        if state.phase == "aggregating":
            return "pause"
        if state.phase == "paused":
            return "pause"
        return "continue"


def get_all_workers() -> list[dict]:
    with get_worker_session() as session:
        workers = session.exec(select(Worker)).all()
        now = time.time()
        result = []
        for w in workers:
            is_stale = (w.last_heartbeat and (now - w.last_heartbeat) > WORKER_TIMEOUT)
            result.append({
                "worker_id": w.worker_id,
                "account_email": w.account_email,
                "assigned_slot": w.assigned_slot,
                "status": "stale" if is_stale else w.status,
                "last_heartbeat": w.last_heartbeat,
                "current_step": w.current_step,
                "loss": w.loss,
                "tokens_processed": w.tokens_processed,
                "gpu_memory_mb": w.gpu_memory_mb,
            })
        return result


def get_active_workers() -> list[str]:
    now = time.time()
    with get_worker_session() as session:
        workers = session.exec(select(Worker)).all()
        return [
            w.worker_id for w in workers
            if w.last_heartbeat and (now - w.last_heartbeat) < WORKER_TIMEOUT
        ]


def should_aggregate(step: int, submitted_workers: set, earliest_submission_time: float = None) -> bool:
    active = set(get_active_workers())
    if not active:
        return False
    if active.issubset(submitted_workers):
        return True
    if earliest_submission_time and (time.time() - earliest_submission_time) > GRADIENT_TIMEOUT:
        return True
    return False


def reap_stale_workers():
    now = time.time()
    with get_worker_session() as session:
        workers = session.exec(select(Worker)).all()
        for w in workers:
            if w.last_heartbeat and (now - w.last_heartbeat) > WORKER_TIMEOUT:
                if w.status != "stale":
                    w.status = "stale"
                    session.add(w)
        session.commit()


def get_worker(worker_id: str):
    with get_worker_session() as session:
        return session.get(Worker, worker_id)
