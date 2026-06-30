import os
from datetime import UTC, datetime

from sqlalchemy import inspect, text
from sqlmodel import Field, Session, SQLModel, create_engine

DB_PATH = os.environ.get("DB_PATH", "/data/cluster.db")
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})


class Worker(SQLModel, table=True):
    worker_id: str = Field(primary_key=True)
    account_email: str
    drive_folder_id: str
    assigned_slot: int
    status: str = "inactive"
    last_heartbeat: float | None = None
    current_step: int = 0
    loss: float | None = None
    tokens_processed: int = 0
    gpu_memory_mb: float | None = None
    created_at: float = Field(default_factory=lambda: datetime.now(UTC).timestamp())
    ip_address: str | None = None


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
    lock_holder: str | None = None
    lock_time: float | None = None
    drive_credentials_json: str | None = None
    oauth_state: str | None = None


def init_db():
    SQLModel.metadata.create_all(engine)
    # create_all() does not add fields to an existing SQLite table. Keep this
    # deliberately small migration here so deployed coordinators upgrade in
    # place instead of requiring users to delete their cluster state.
    existing = {column["name"] for column in inspect(engine).get_columns("clusterstate")}
    additions = {
        "drive_credentials_json": "TEXT",
        "oauth_state": "TEXT",
    }
    with engine.begin() as connection:
        for column, sql_type in additions.items():
            if column not in existing:
                connection.execute(text(f"ALTER TABLE clusterstate ADD COLUMN {column} {sql_type}"))


def get_worker_session():
    return Session(engine)
