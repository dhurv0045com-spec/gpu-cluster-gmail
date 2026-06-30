import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException
from google.oauth2.credentials import Credentials

from backend import auth, main


class FakeDriveSync:
    def __init__(self, credentials, folder_id):
        self.credentials = credentials
        self.folder_id = folder_id
        self.written_state = None
        self.service = self

    def write_coordinator_state(self, state):
        self.written_state = state.copy()

    def file_exists(self, filename):
        return filename == "anra.pt"

    def files(self):
        return self

    def list(self, **kwargs):
        return self

    def execute(self):
        return {"files": []}


def credential_dict():
    return auth.credentials_to_dict(
        Credentials(
            token="realistic-access-token",
            refresh_token="realistic-refresh-token",
            token_uri="https://oauth2.googleapis.com/token",
            client_id="client-id.apps.googleusercontent.com",
            client_secret="client-secret",
            scopes=auth.SCOPES,
            expiry=datetime.now(UTC) + timedelta(hours=1),
        )
    )


def test_cluster_init_requires_stored_credentials(isolated_database, monkeypatch):
    monkeypatch.setattr(main, "DriveSync", FakeDriveSync)
    request = main.ClusterInitRequest(
        coordinator_drive_folder_id="folder-123",
        master_checkpoint_filename="anra.pt",
        total_target_steps=100,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main.cluster_init(request))
    assert exc_info.value.status_code == 401


def test_stored_credentials_initialize_a_real_drive_client(isolated_database, monkeypatch):
    monkeypatch.setattr(main, "DriveSync", FakeDriveSync)
    auth.store_credentials(credential_dict())
    loaded = auth.load_credentials(refresh=False)
    assert loaded is not None and loaded.valid

    result = asyncio.run(
        main.cluster_init(
            main.ClusterInitRequest(
                coordinator_drive_folder_id="folder-123",
                master_checkpoint_filename="anra.pt",
                total_target_steps=100,
            )
        )
    )

    assert result["status"] == "initialized"
    assert main.drive_sync_instance.credentials.token == "realistic-access-token"
    assert main.drive_sync_instance.files().list().execute() == {"files": []}
    assert main.drive_sync_instance.written_state["delta_history"] == []
