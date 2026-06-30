import io
import json
import logging
import os
import tempfile
import time

from google.auth.transport.requests import Request as AuthRequest
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger("anra-coordinator.drive")

LOCK_TIMEOUT = 30
COORDINATOR_STATE_FILE = "coordinator_state.json"
MAX_RETRIES = 5

API_ERRORS = (ConnectionError, TimeoutError, IOError)


class DriveSync:
    def __init__(self, credentials: Credentials, cluster_folder_id: str):
        self._credentials = credentials
        self.folder_id = cluster_folder_id
        self._service = None

    @property
    def service(self):
        if self._service is None:
            self._refresh_credentials()
            self._service = build("drive", "v3", credentials=self._credentials)
        return self._service

    def _refresh_credentials(self):
        if self._credentials and self._credentials.expired and self._credentials.refresh_token:
            try:
                self._credentials.refresh(AuthRequest())
                logger.info("Drive credentials refreshed")
            except Exception as e:
                logger.warning(f"Failed to refresh credentials: {e}")

    def update_credentials(self, credentials: Credentials):
        self._credentials = credentials
        self._service = None

    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=2, max=30),
           retry=retry_if_exception_type(API_ERRORS))
    def _list_files(self, query: str, fields: str = "files(id, name, mimeType, size, createdTime, parents)"):
        all_files = []
        page_token = None
        while True:
            response = self.service.files().list(
                q=query, spaces="drive", fields=f"nextPageToken, {fields}",
                pageToken=page_token, pageSize=200,
            ).execute()
            all_files.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return all_files

    def _get_file_id(self, filename: str, parent_id: str | None = None) -> str | None:
        parent = parent_id or self.folder_id
        query = f"name='{filename}' and '{parent}' in parents and trashed=false"
        items = self._list_files(query, "files(id, name)")
        return items[0]["id"] if items else None

    def _ensure_folder(self, name: str, parent_id: str | None = None) -> str:
        parent = parent_id or self.folder_id
        existing = self._get_file_id(name, parent)
        if existing:
            return existing
        metadata = {"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent]}
        folder = self.service.files().create(body=metadata, fields="id").execute()
        return folder["id"]

    def file_exists(self, filename: str) -> bool:
        return self._get_file_id(filename) is not None

    def acquire_lock(self, holder_id: str, timeout: int = LOCK_TIMEOUT) -> bool:
        current = self.read_coordinator_state()
        lock_holder = current.get("lock_holder")
        lock_time = current.get("lock_time", 0)
        if lock_holder is not None and (time.time() - lock_time) < timeout and lock_holder != holder_id:
            return False
        current["lock_holder"] = holder_id
        current["lock_time"] = time.time()
        self.write_coordinator_state(current)
        time.sleep(2)
        verify = self.read_coordinator_state()
        won = verify.get("lock_holder") == holder_id
        if won:
            logger.debug(f"Lock acquired by {holder_id}")
        else:
            logger.warning(f"Lock contention: {holder_id} lost to {verify.get('lock_holder')}")
        return won

    def release_lock(self, holder_id: str):
        current = self.read_coordinator_state()
        if current.get("lock_holder") == holder_id:
            current["lock_holder"] = None
            current["lock_time"] = None
            self.write_coordinator_state(current)
            logger.debug(f"Lock released by {holder_id}")

    def write_master_weights(self, local_path: str, version: int) -> str:
        filename = f"master_weights_v{version}.pt"
        existing = self._get_file_id(filename)
        media = MediaFileUpload(local_path, resumable=True)
        if existing:
            file = self.service.files().update(fileId=existing, media_body=media).execute()
        else:
            metadata = {"name": filename, "parents": [self.folder_id]}
            file = self.service.files().create(body=metadata, media_body=media, fields="id").execute()
        logger.info(f"Uploaded {filename} ({os.path.getsize(local_path) / 1024 / 1024:.1f} MB)")
        return file["id"]

    def write_sparse_delta(self, local_path: str, version: int) -> str:
        filename = f"delta_v{version:06d}.npz"
        existing = self._get_file_id(filename)
        media = MediaFileUpload(local_path, mimetype="application/octet-stream", resumable=True)
        if existing:
            self.service.files().update(fileId=existing, media_body=media, fields="id").execute()
        else:
            metadata = {"name": filename, "parents": [self.folder_id]}
            self.service.files().create(body=metadata, media_body=media, fields="id").execute()
        logger.info(f"Uploaded {filename} ({os.path.getsize(local_path) / 1024 / 1024:.1f} MB)")
        return filename

    def download_file(self, filename: str, local_path: str | None = None) -> str:
        file_id = self._get_file_id(filename)
        if not file_id:
            raise FileNotFoundError(f"Drive file {filename} not found")
        destination = local_path or os.path.join(tempfile.gettempdir(), filename)
        request = self.service.files().get_media(fileId=file_id)
        with io.FileIO(destination, "wb") as file_handle:
            downloader = MediaIoBaseDownload(file_handle, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        return destination

    def read_gradient_file(self, worker_id: str, step: int) -> str:
        worker_folder_id = self._get_file_id(worker_id)
        if not worker_folder_id:
            raise FileNotFoundError(f"Worker folder {worker_id} not found")
        filename = f"grad_step_{step:06d}.npz"
        file_id = self._get_file_id(filename, worker_folder_id)
        if not file_id:
            raise FileNotFoundError(f"Gradient file {filename} not found for {worker_id}")
        local_path = os.path.join(tempfile.gettempdir(), f"{worker_id}_grad_step_{step:06d}.npz")
        request = self.service.files().get_media(fileId=file_id)
        with io.FileIO(local_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        return local_path

    def write_coordinator_state(self, state: dict):
        content = json.dumps(state, indent=2, default=str)
        local_path = os.path.join(tempfile.gettempdir(), COORDINATOR_STATE_FILE)
        with open(local_path, "w") as f:
            f.write(content)
        existing = self._get_file_id(COORDINATOR_STATE_FILE)
        media = MediaFileUpload(local_path, mimetype="application/json")
        if existing:
            self.service.files().update(fileId=existing, media_body=media).execute()
        else:
            metadata = {"name": COORDINATOR_STATE_FILE, "parents": [self.folder_id]}
            self.service.files().create(body=metadata, media_body=media, fields="id").execute()

    def read_coordinator_state(self) -> dict:
        file_id = self._get_file_id(COORDINATOR_STATE_FILE)
        if not file_id:
            return {
                "global_step": 0, "phase": "idle", "expected_workers": [],
                "submitted_this_step": [], "master_weights_version": 0,
                "master_weights_path": "", "current_lr": 3e-4,
                "checkpoint_version": 0, "delta_path": "", "delta_history": [],
                "latest_layer_norms": [],
                "lock_holder": None, "lock_time": None,
                "total_target_steps": 0, "submission_times": {},
                "gradient_submissions": {},
            }
        request = self.service.files().get_media(fileId=file_id)
        content = io.BytesIO()
        downloader = MediaIoBaseDownload(content, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        try:
            return json.loads(content.getvalue().decode())
        except json.JSONDecodeError:
            logger.warning("Corrupt coordinator_state.json, returning defaults")
            return {
                "global_step": 0, "phase": "idle", "expected_workers": [],
                "submitted_this_step": [], "master_weights_version": 0,
                "master_weights_path": "", "current_lr": 3e-4,
                "checkpoint_version": 0, "delta_path": "", "delta_history": [],
                "latest_layer_norms": [],
                "lock_holder": None, "lock_time": None,
                "total_target_steps": 0, "submission_times": {},
                "gradient_submissions": {},
            }

    def list_submitted_gradients(self, step: int, worker_ids: list[str]) -> list[str]:
        found = []
        for worker_id in worker_ids:
            file_id = self.list_worker_gradient_files(worker_id, step)
            if file_id:
                found.append(file_id)
        return found

    def list_worker_gradient_files(self, worker_id: str, step: int) -> str | None:
        worker_folder_id = self._get_file_id(worker_id)
        if not worker_folder_id:
            return None
        filename = f"grad_step_{step:06d}.npz"
        return self._get_file_id(filename, worker_folder_id)

    def cleanup_old_gradients(self, keep_last_n_steps: int = 3):
        all_worker_folders = self._list_files(
            f"'{self.folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
            "files(id, name)",
        )
        for folder in all_worker_folders:
            folder_id = folder["id"]
            grad_files = self._list_files(
                f"'{folder_id}' in parents and name contains 'grad_step_' and trashed=false",
                "files(id, name, createdTime)",
            )
            grad_files.sort(key=lambda f: f.get("createdTime", ""), reverse=True)
            for f in grad_files[keep_last_n_steps:]:
                try:
                    self.service.files().delete(fileId=f["id"]).execute()
                    logger.debug(f"Deleted old gradient: {f['name']}")
                except Exception:
                    pass

    def list_all_files(self):
        return self._list_files(f"'{self.folder_id}' in parents and trashed=false")
