"""FastAPI service: anomaly detection copilot for renewable-asset telemetry."""
import io
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .copilot import answer
from .pipeline import (CONFIG, extract_events, load_data, run_pipeline, series_payload, summarize)

app = FastAPI(title="Renewable Asset Fault & Anomaly Detection Copilot", version="1.0.0")
BASE = Path(__file__).resolve().parents[1]
FRONTEND = BASE / "frontend"
MAX_UPLOAD = 2_000_000  # bytes


class Query(BaseModel):
    text: str = ""


@lru_cache(maxsize=1)
def _bundled():
    f = run_pipeline()
    return f, extract_events(f)


def _bundle(f, events):
    return dict(summary=summarize(f, events), events=events,
                assets=sorted(f.asset_id.unique().tolist()))


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/demo")
def demo():
    f, ev = _bundled()
    return dict(case_study="Renewable Asset Fault & Anomaly Detection Copilot", status="running",
                config=CONFIG, **_bundle(f, ev))


@app.get("/api/analyze")
def analyze(asset_id: str):
    f, ev = _bundled()
    if asset_id not in set(f.asset_id):
        raise HTTPException(404, f"Unknown asset_id '{asset_id}'")
    return dict(asset_id=asset_id, series=series_payload(f, asset_id),
                events=[e for e in ev if e["asset_id"] == asset_id])


@app.post("/api/analyze_upload")
async def analyze_upload(file: UploadFile = File(...)):
    """Run the same pipeline on an uploaded CSV with the bundled schema (not cached)."""
    raw = await file.read()
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "File too large (max 2 MB)")
    try:
        f = run_pipeline(load_data(pd.read_csv(io.BytesIO(raw))))
    except Exception as e:
        raise HTTPException(422, f"Could not analyse file: {e}")
    ev = extract_events(f)
    return dict(**_bundle(f, ev), series={a: series_payload(f, a) for a in f.asset_id.unique()})


@app.get("/api/evaluation")
def evaluation():
    p = BASE / "evaluation_results.json"
    if not p.exists():
        raise HTTPException(404, "Run `python -m backend.evaluate` first")
    return json.loads(p.read_text())


@app.post("/api/run")
def run_case(query: Query):
    """Copilot Q&A: deterministic intent routing over the pipeline output (no external LLM needed)."""
    f, ev = _bundled()
    evaluation = None
    p = BASE / "evaluation_results.json"
    if p.exists():
        evaluation = json.loads(p.read_text())
    return answer(query.text, sorted(f.asset_id.unique().tolist()), ev, evaluation)


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")