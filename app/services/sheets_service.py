import json
import io

import pandas as pd
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

from app.config import settings

_credentials = None
_drive_service = None
_sheets_service = None


def _get_credentials():
    global _credentials
    if _credentials is None:
        info = json.loads(settings.google_credentials_json)
        _credentials = Credentials.from_service_account_info(
            info,
            scopes=[
                "https://www.googleapis.com/auth/drive.readonly",
                "https://www.googleapis.com/auth/spreadsheets.readonly",
            ],
        )
    return _credentials


def _get_drive_service():
    global _drive_service
    if _drive_service is None:
        _drive_service = build("drive", "v3", credentials=_get_credentials())
    return _drive_service


def _get_sheets_service():
    global _sheets_service
    if _sheets_service is None:
        _sheets_service = build("sheets", "v4", credentials=_get_credentials())
    return _sheets_service


def get_sheets_from_folder() -> dict:
    """Get all Google Sheets from the configured Drive folder.
    Returns dict of {name: {id, url, webViewLink}}."""
    service = _get_drive_service()
    folder_id = settings.google_drive_folder_id
    query = (
        f"'{folder_id}' in parents and "
        "mimeType='application/vnd.google-apps.spreadsheet' and trashed=false"
    )
    results = service.files().list(
        q=query, pageSize=100, fields="files(id, name, webViewLink)"
    ).execute()

    sheets = {}
    for item in results.get("files", []):
        sheets[item["name"]] = {
            "id": item["id"],
            "url": f"https://docs.google.com/spreadsheets/d/{item['id']}/edit",
            "webViewLink": item.get("webViewLink", ""),
        }
    return dict(sorted(sheets.items()))


def read_sheet(spreadsheet_id: str) -> pd.DataFrame:
    """Read a Google Sheet into a DataFrame using the Sheets API."""
    service = _get_sheets_service()

    # Get all sheet data (first sheet)
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id, range="A:ZZ"
    ).execute()
    values = result.get("values", [])

    if not values:
        return pd.DataFrame()

    # First row is header
    headers = values[0]
    rows = values[1:]

    # Pad rows to match header length
    for i, row in enumerate(rows):
        if len(row) < len(headers):
            rows[i] = row + [""] * (len(headers) - len(row))
        elif len(row) > len(headers):
            rows[i] = row[: len(headers)]

    # Deduplicate column names (e.g., two "PARG Number" columns)
    seen = {}
    deduped = []
    for h in headers:
        if h in seen:
            seen[h] += 1
            deduped.append(f"{h}.{seen[h]}")
        else:
            seen[h] = 0
            deduped.append(h)
    headers = deduped

    df = pd.DataFrame(rows, columns=headers)

    # Convert numeric-looking columns
    for col in df.columns:
        if not isinstance(df[col], pd.Series):
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any() and converted.notna().sum() >= df[col].notna().sum() * 0.5:
            df[col] = converted

    return df


def get_column_descriptions() -> dict[str, str]:
    """Load column descriptions from *_columns sheets in Google Drive.

    Each _columns sheet has headers "Column Name" and "Description".
    """
    sheets = get_sheets_from_folder()
    descs: dict[str, str] = {}
    for name, info in sheets.items():
        if not name.endswith("_columns"):
            continue
        try:
            df = read_sheet(info["id"])
            if "Column Name" in df.columns and "Description" in df.columns:
                for _, row in df.iterrows():
                    col_name = str(row["Column Name"]).strip()
                    col_desc = str(row["Description"]).strip()
                    if col_name and col_desc and col_name != "nan" and col_desc != "nan":
                        descs[col_name] = col_desc
        except Exception:
            continue

    return descs
