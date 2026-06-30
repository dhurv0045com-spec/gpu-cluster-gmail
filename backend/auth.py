import json
import os
from datetime import UTC

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from sqlmodel import Session

from .database import ClusterState, engine

SCOPES = [
    # The coordinator writes into a folder chosen by ID rather than through a
    # Google Picker. drive.file cannot reliably access such pre-existing shared
    # folders, so the full Drive scope is required for this deployment model.
    "https://www.googleapis.com/auth/drive",
]
CLIENT_SECRETS_FILE = os.environ.get("GOOGLE_CLIENT_SECRETS", "client_secret.json")
REDIRECT_URI = os.environ.get("REDIRECT_URI", "http://localhost:8000/api/auth/callback")


def get_oauth_flow() -> Flow:
    flow = Flow.from_client_secrets_file(CLIENT_SECRETS_FILE, scopes=SCOPES)
    flow.redirect_uri = REDIRECT_URI
    return flow


def get_authorization_url() -> tuple[str, str]:
    flow = get_oauth_flow()
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    return auth_url, state


def exchange_code_for_token(code: str) -> Credentials:
    flow = get_oauth_flow()
    flow.fetch_token(code=code)
    return flow.credentials


def credentials_to_dict(credentials: Credentials) -> dict:
    result = {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": credentials.scopes,
    }
    if credentials.expiry is not None:
        expiry = credentials.expiry
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=UTC)
        result["expiry"] = expiry.isoformat()
    return result


def dict_to_credentials(creds_dict: dict) -> Credentials:
    return Credentials.from_authorized_user_info(creds_dict, SCOPES)


def store_credentials(creds_dict: dict) -> None:
    with Session(engine) as session:
        state = session.get(ClusterState, 1)
        if state is None:
            state = ClusterState(id=1)
        state.drive_credentials_json = json.dumps(creds_dict)
        session.add(state)
        session.commit()


def load_credentials(refresh: bool = True) -> Credentials | None:
    with Session(engine) as session:
        state = session.get(ClusterState, 1)
        if state is None or not state.drive_credentials_json:
            return None
        credentials = dict_to_credentials(json.loads(state.drive_credentials_json))

    if refresh and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        store_credentials(credentials_to_dict(credentials))
    return credentials


def store_oauth_state(oauth_state: str) -> None:
    with Session(engine) as session:
        state = session.get(ClusterState, 1)
        if state is None:
            state = ClusterState(id=1)
        state.oauth_state = oauth_state
        session.add(state)
        session.commit()


def consume_oauth_state(oauth_state: str) -> bool:
    """Validate a callback once, then invalidate it to prevent replay."""
    with Session(engine) as session:
        state = session.get(ClusterState, 1)
        if state is None or not state.oauth_state or state.oauth_state != oauth_state:
            return False
        state.oauth_state = None
        session.add(state)
        session.commit()
        return True
