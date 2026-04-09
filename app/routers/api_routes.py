"""JSON API endpoints for data (used by AG Grid) and column metadata."""
import json
import math

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models import User, DatasetSelection
from app.services.merge_service import load_dataset
from app.services.filter_service import apply_filters
from app.services.sheets_service import get_column_descriptions

router = APIRouter(prefix="/api")


@router.get("/data")
async def get_data(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    dataset_key: str = Query(...),
):
    """Return dataset as JSON for AG Grid. Applies per-user filters."""
    try:
        df = load_dataset(dataset_key)
    except Exception as e:
        return JSONResponse({"columns": [], "rows": [], "error": str(e)})

    # Apply per-user filters
    sel = db.query(DatasetSelection).filter(
        DatasetSelection.user_id == user.id,
        DatasetSelection.dataset_key == dataset_key,
    ).first()
    if sel and sel.filters_json:
        filters = json.loads(sel.filters_json)
        if filters:
            df = apply_filters(df, filters)

    columns = df.columns.tolist()
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

    # Include column descriptions for header tooltips
    col_descriptions = {col: get_column_descriptions().get(col, "") for col in columns}

    return JSONResponse({"columns": columns, "rows": rows, "column_descriptions": col_descriptions})
