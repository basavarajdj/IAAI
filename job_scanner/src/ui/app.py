from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.apply.applier import JobApplier
from src.llm.ollama_client import OllamaClient
from src.resume.parser import load_resume_profile
from src.settings import EnvSettings, load_yaml_config, resolve_resume_path
from src.storage.job_store import JobStore

UI_DIR = Path(__file__).resolve().parent / "static"


class MarkAppliedBody(BaseModel):
    message: str = "Marked applied via UI"


def create_app(db_path: Optional[Path] = None) -> FastAPI:
    env = EnvSettings()
    yaml_cfg = load_yaml_config()
    output_cfg = yaml_cfg.get("output", {})
    db = db_path or Path(output_cfg.get("results_dir", "output")) / "jobs.db"

    store = JobStore(db)
    matches_json = Path(output_cfg.get("results_dir", "output")) / "matches.json"
    if store.stats()["total"] == 0 and matches_json.exists():
        store.import_matches_json(matches_json)

    app = FastAPI(title="Job Scanner", version="0.2.0")

    @app.get("/api/stats")
    def stats():
        return store.stats()

    @app.get("/api/jobs")
    def list_jobs(
        portal: Optional[str] = None,
        applied: Optional[bool] = None,
        min_score: int = Query(0, ge=0, le=100),
        search: str = "",
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ):
        return store.list_jobs(
            portal=portal,
            applied=applied,
            min_score=min_score,
            search=search,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = store.get_job(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return job

    @app.post("/api/jobs/{job_id}/mark-applied")
    def mark_applied(job_id: str, body: MarkAppliedBody):
        job = store.get_job(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if store.is_applied(job_id):
            return {"ok": True, "message": "Already marked as applied"}
        store.mark_manual_applied(job_id, body.message)
        return {"ok": True, "message": "Marked as applied"}

    @app.post("/api/jobs/{job_id}/apply")
    def apply_job(job_id: str):
        if store.is_applied(job_id):
            raise HTTPException(409, "Already applied to this job")

        match = store.match_result_for_job(job_id)
        if not match:
            raise HTTPException(404, "Job not found")

        llm = OllamaClient(env.ollama_host, env.ollama_model)
        profile = load_resume_profile(resolve_resume_path(env), llm)
        log_path = Path(output_cfg.get("applications_log", "output/applications.jsonl"))
        applier = JobApplier(llm, env, log_path, store=store)
        record = applier.apply_single(match, profile)

        if record.status == "skipped":
            raise HTTPException(409, record.message)
        return record.model_dump(mode="json")

    @app.get("/")
    def index():
        return FileResponse(UI_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")
    return app
