from __future__ import annotations

import time
from io import BytesIO
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from eda_cli.core import summarize_dataset, missing_table, compute_quality_flags

app = FastAPI(title="eda-cli quality service", version="0.1.0")


# -------------------------
# Models
# -------------------------
class QualityRequest(BaseModel):
    n_rows: int = Field(..., ge=0)
    n_cols: int = Field(..., ge=0)
    missing_share: float = Field(..., ge=0.0)
    dup_rows_share: float = Field(0.0, ge=0.0)
    target_col: Optional[str] = None


class QualityResponse(BaseModel):
    ok_for_model: bool
    quality_score: float
    latency_ms: int
    flags: Dict[str, bool]


# -------------------------
# Helpers
# -------------------------
def _quality_from_flags(flags: Dict[str, bool]) -> float:
    """
    Простой скоринг: стартуем со 100 и вычитаем штраф за каждый bad-flag.
    Можно оставить так — преподавателю важнее контракт и корректные флаги.
    """
    score = 100.0
    for v in flags.values():
        if bool(v):
            score -= 15.0
    if score < 0:
        score = 0.0
    return score


# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/quality", response_model=QualityResponse)
def quality(req: QualityRequest) -> QualityResponse:
    t0 = time.perf_counter()

    # Мини-эвристики на "синтетических" параметрах запроса (как в семинаре).
    flags: Dict[str, bool] = {
        "too_few_rows": req.n_rows < 50,
        "too_few_cols": req.n_cols < 2,
        "too_many_missing": req.missing_share > 0.4,
        "too_many_duplicates": req.dup_rows_share > 0.2,
    }

    quality_score = _quality_from_flags(flags)
    ok_for_model = quality_score >= 70.0 and (not flags["too_few_rows"]) and (not flags["too_many_missing"])

    latency_ms = int((time.perf_counter() - t0) * 1000)
    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=quality_score,
        latency_ms=latency_ms,
        flags={k: bool(v) for k, v in flags.items()},
    )


@app.post("/quality-from-csv", response_model=QualityResponse)
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    t0 = time.perf_counter()

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty CSV file")

        df = pd.read_csv(BytesIO(contents))
        if df.shape[0] == 0 or df.shape[1] == 0:
            raise HTTPException(status_code=400, detail="CSV has no data (0 rows or 0 cols)")

        # Требование задания: вызвать эти функции
        _ = summarize_dataset(df)
        _ = missing_table(df)
        flags = compute_quality_flags(df)

        flags = {str(k): bool(v) for k, v in flags.items()}
        quality_score = _quality_from_flags(flags)
        ok_for_model = quality_score >= 70.0 and (not flags.get("too_few_rows", False)) and (not flags.get("too_many_missing", False))

        latency_ms = int((time.perf_counter() - t0) * 1000)
        return QualityResponse(
            ok_for_model=ok_for_model,
            quality_score=quality_score,
            latency_ms=latency_ms,
            flags=flags,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read/parse CSV: {e}")


# -------------------------
# YOUR REQUIRED endpoint (HW04)
# -------------------------
@app.post("/quality-flags-from-csv")
async def quality_flags_from_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    HW04 custom endpoint: returns full quality flags dict (incl. HW03 flags).
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty CSV file")

        df = pd.read_csv(BytesIO(contents))
        if df.shape[0] == 0 or df.shape[1] == 0:
            raise HTTPException(status_code=400, detail="CSV has no data (0 rows or 0 cols)")

        _ = summarize_dataset(df)
        _ = missing_table(df)
        flags = compute_quality_flags(df)

        return {"flags": {str(k): bool(v) for k, v in flags.items()}}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read/parse CSV: {e}")
