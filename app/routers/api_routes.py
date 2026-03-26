"""JSON API endpoints for data (used by AG Grid) and column metadata."""
import math
from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models import User
from app.services.merge_service import load_data
from app.services.filter_service import apply_filters

router = APIRouter(prefix="/api")


@router.get("/data")
async def get_data(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    sheet_id: str = Query(None),
):
    prefs = user.preferences
    if not prefs:
        return JSONResponse({"columns": [], "rows": []})

    try:
        df = load_data(prefs.data_source_type, sheet_id)
    except Exception as e:
        return JSONResponse({"columns": [], "rows": [], "error": str(e)})

    # Apply filters for PEITHO merge
    if prefs.data_source_type == "veev_peitho_merge":
        df = apply_filters(df, prefs.chi2_max, prefs.rmse_max)

    columns = df.columns.tolist()

    # Convert to JSON-safe records
    rows = []
    for _, row in df.iterrows():
        record = {}
        for col in columns:
            val = row[col]
            if val is None or (isinstance(val, float) and math.isnan(val)):
                record[col] = None
            else:
                record[col] = val
        rows.append(record)

    return JSONResponse({"columns": columns, "rows": rows})


@router.get("/columns")
async def get_columns(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    prefs = user.preferences
    if not prefs:
        return JSONResponse({"columns": []})

    df = load_data(prefs.data_source_type)
    return JSONResponse({"columns": df.columns.tolist()})
