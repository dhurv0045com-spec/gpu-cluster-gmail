import os
import json
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from sqlmodel import Session, select
from database import engine, ClusterState

SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive.metadata.readonly",
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
    return {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": credentials.scopes,
    }


def dict_to_credentials(creds_dict: dict) -> Credentials:
    return Credentials.from_authorized_user_info(creds_dict, SCOPES)


def store_credentials(user_id: str, creds_dict: dict):
    with Session(engine) as session:
        state = session.get(ClusterState, 1)
        if state:
            setattr(state, f"drive_credentials_{user_id}", json.dumps(creds_dict))
            session.add(state)
            session.commit()
