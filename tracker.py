"""Tiny local application tracker backed by a JSON file."""
import json
import uuid
from datetime import date
from pathlib import Path

DB_PATH = Path(__file__).parent / "jobs.json"
STATUSES = ["saved", "applied", "interviewing", "offer", "rejected", "withdrawn"]


def _load() -> list[dict]:
    if not DB_PATH.exists():
        return []
    return json.loads(DB_PATH.read_text())


def _save(jobs: list[dict]) -> None:
    DB_PATH.write_text(json.dumps(jobs, indent=2))


def add_job(company: str, role: str, url: str = "", fit_score: int | None = None,
            notes: str = "") -> dict:
    jobs = _load()
    job = {
        "id": uuid.uuid4().hex[:6],
        "company": company,
        "role": role,
        "url": url,
        "fit_score": fit_score,
        "status": "saved",
        "notes": notes,
        "added": date.today().isoformat(),
        "updated": date.today().isoformat(),
    }
    jobs.append(job)
    _save(jobs)
    return job


def list_jobs(status: str | None = None) -> list[dict]:
    jobs = _load()
    if status:
        jobs = [j for j in jobs if j["status"] == status]
    return sorted(jobs, key=lambda j: (j["fit_score"] or 0), reverse=True)


def update_job(job_id: str, status: str | None = None, notes: str | None = None) -> dict | None:
    if status is not None and status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    jobs = _load()
    for job in jobs:
        if job["id"] == job_id:
            if status is not None:
                job["status"] = status
            if notes:
                job["notes"] = (job["notes"] + "\n" + notes).strip()
            job["updated"] = date.today().isoformat()
            _save(jobs)
            return job
    return None
