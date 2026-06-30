import pytest
from sqlmodel import SQLModel, create_engine

from backend import auth, database, main


@pytest.fixture
def isolated_database(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'cluster.db'}", connect_args={"check_same_thread": False}
    )
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(auth, "engine", engine)
    SQLModel.metadata.create_all(engine)
    main.drive_sync_instance = None
    main.aggregation_in_progress = False
    yield engine
    main.drive_sync_instance = None
    main.aggregation_in_progress = False
    engine.dispose()
