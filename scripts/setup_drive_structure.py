import os
import sys
import json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


def setup_drive_structure(credentials: Credentials, root_folder_name: str = "AnRa") -> dict:
    service = build("drive", "v3", credentials=credentials)

    def find_or_create_folder(name, parent_id=None):
        query = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if parent_id:
            query += f" and '{parent_id}' in parents"
        results = service.files().list(q=query, fields="files(id, name)", pageSize=1).execute()
        items = results.get("files", [])
        if items:
            return items[0]["id"]
        metadata = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
        if parent_id:
            metadata["parents"] = [parent_id]
        folder = service.files().create(body=metadata, fields="id").execute()
        return folder["id"]

    print(f"Creating Drive structure under '{root_folder_name}'...")
    root_id = find_or_create_folder(root_folder_name)
    cluster_id = find_or_create_folder("cluster", root_id)

    initial_state = {
        "global_step": 0,
        "phase": "idle",
        "expected_workers": [],
        "submitted_this_step": [],
        "master_weights_version": 0,
        "master_weights_path": "",
        "current_lr": 3e-4,
        "lock_holder": None,
        "lock_time": None,
        "total_target_steps": 0,
    }
    import tempfile
    state_path = os.path.join(tempfile.gettempdir(), "coordinator_state.json")
    with open(state_path, "w") as f:
        json.dump(initial_state, f, indent=2)
    from googleapiclient.http import MediaFileUpload
    state_file = service.files().create(
        body={"name": "coordinator_state.json", "parents": [cluster_id]},
        media_body=MediaFileUpload(state_path, mimetype="application/json"),
        fields="id",
    ).execute()

    print(f"Root folder ID: {root_id}")
    print(f"Cluster folder ID: {cluster_id}")
    print(f"State file ID: {state_file['id']}")
    print("\nIMPORTANT: Share the cluster folder with each worker's Gmail account:")
    print(f"  1. Go to https://drive.google.com/drive/folders/{cluster_id}")
    print("  2. Share with each worker email (Editor role)")
    return {"root_folder_id": root_id, "cluster_folder_id": cluster_id}


if __name__ == "__main__":
    from google_auth_oauthlib.flow import InstalledAppFlow
    SCOPES = ["https://www.googleapis.com/auth/drive.file"]
    flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
    creds = flow.run_local_server(port=0)
    result = setup_drive_structure(creds)
    print(f"\nSave this cluster_folder_id for the web app setup: {result['cluster_folder_id']}")
