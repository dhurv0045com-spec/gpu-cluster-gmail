import os
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field, Session, create_engine
from typing import Optional

DB_PATH = os.environ.get("DB_PATH", "/data/cluster.db")
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})


class Worker(SQLModel, table=True):
    worker_id: str = Field(primary_key=True)
    account_email: str
    drive_folder_id: str
    assigned_slot: int
    status: str = "inactive"
    last_heartbeat: Optional[float] = None
    current_step: int = 0
    loss: Optional[float] = None
    tokens_processed: int = 0
    gpu_memory_mb: Optional[float] = None
    created_at: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    ip_address: Optional[str] = None


class ClusterState(SQLModel, table=True):
    id: int = Field(default=1, primary_key=True)
    coordinator_drive_folder_id: str = ""
    master_checkpoint_filename: str = ""
    total_target_steps: int = 0
    global_step: int = 0
    phase: str = "idle"
    master_weights_version: int = 0
    master_weights_path: str = ""
    current_lr: float = 3e-4
    total_loss_history: str = ""
    tokens_per_second_total: float = 0.0
    estimated_eta: float = 0.0
    lock_holder: Optional[str] = None
    lock_time: Optional[float] = None


def init_db():
    SQLModel.metadata.create_all(engine)


def get_worker_session():
    return Session(engine)
