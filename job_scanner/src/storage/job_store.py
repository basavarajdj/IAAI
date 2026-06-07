from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.models import ApplicationRecord, JobListing, MatchResult, Portal
from src.storage.database import get_connection
from src.storage.keys import normalize_job_key

APPLIED_STATUSES = frozenset({"submitted", "assisted", "manual"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn = get_connection(db_path)

    def close(self) -> None:
        self.conn.close()

    def upsert_match(self, match: MatchResult) -> str:
        job = match.job
        pk = normalize_job_key(job.portal, job.job_id, job.url)
        now = _now()
        scraped = job.scraped_at.isoformat() if hasattr(job.scraped_at, "isoformat") else str(job.scraped_at)

        self.conn.execute(
            """
            INSERT INTO jobs (
                id, portal, job_id, title, company, location, url, description, posted,
                scraped_at, match_score, reasoning, skill_overlap, gaps, recommended,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                company = excluded.company,
                location = excluded.location,
                url = excluded.url,
                description = CASE WHEN length(excluded.description) > length(jobs.description)
                    THEN excluded.description ELSE jobs.description END,
                posted = COALESCE(excluded.posted, jobs.posted),
                scraped_at = excluded.scraped_at,
                match_score = excluded.match_score,
                reasoning = excluded.reasoning,
                skill_overlap = excluded.skill_overlap,
                gaps = excluded.gaps,
                recommended = excluded.recommended,
                updated_at = excluded.updated_at
            """,
            (
                pk,
                job.portal.value,
                job.job_id,
                job.title,
                job.company,
                job.location,
                job.url,
                job.description,
                job.posted,
                scraped,
                match.score,
                match.reasoning,
                json.dumps(match.skill_overlap),
                json.dumps(match.gaps),
                1 if match.recommended else 0,
                now,
                now,
            ),
        )
        self.conn.commit()
        return pk

    def upsert_job_listing(self, job: JobListing) -> str:
        pk = normalize_job_key(job.portal, job.job_id, job.url)
        now = _now()
        scraped = job.scraped_at.isoformat() if hasattr(job.scraped_at, "isoformat") else str(job.scraped_at)

        self.conn.execute(
            """
            INSERT INTO jobs (
                id, portal, job_id, title, company, location, url, description, posted,
                scraped_at, match_score, reasoning, skill_overlap, gaps, recommended,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, '', '[]', '[]', 0, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                company = excluded.company,
                location = excluded.location,
                url = excluded.url,
                description = CASE WHEN length(excluded.description) > length(jobs.description)
                    THEN excluded.description ELSE jobs.description END,
                scraped_at = excluded.scraped_at,
                updated_at = excluded.updated_at
            """,
            (
                pk,
                job.portal.value,
                job.job_id,
                job.title,
                job.company,
                job.location,
                job.url,
                job.description,
                job.posted,
                scraped,
                now,
                now,
            ),
        )
        self.conn.commit()
        return pk

    def save_matches(self, matches: list[MatchResult]) -> list[str]:
        return [self.upsert_match(m) for m in matches]

    def update_description(self, job_pk: str, description: str) -> None:
        self.conn.execute(
            "UPDATE jobs SET description = ?, updated_at = ? WHERE id = ?",
            (description, _now(), job_pk),
        )
        self.conn.commit()

    def is_applied(self, job_pk: str) -> bool:
        row = self.conn.execute(
            "SELECT status FROM applications WHERE job_pk = ?",
            (job_pk,),
        ).fetchone()
        return row is not None and row["status"] in APPLIED_STATUSES

    def get_application(self, job_pk: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM applications WHERE job_pk = ?",
            (job_pk,),
        ).fetchone()
        return dict(row) if row else None

    def record_application(self, job_pk: str, record: ApplicationRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO applications (job_pk, status, message, applied_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(job_pk) DO UPDATE SET
                status = excluded.status,
                message = excluded.message,
                applied_at = excluded.applied_at
            """,
            (
                job_pk,
                record.status,
                record.message,
                record.applied_at.isoformat()
                if hasattr(record.applied_at, "isoformat")
                else str(record.applied_at),
            ),
        )
        self.conn.commit()

    def mark_manual_applied(self, job_pk: str, message: str = "Marked applied via UI") -> None:
        rec = ApplicationRecord(
            job_url="",
            portal=Portal.LINKEDIN,
            title="",
            company="",
            match_score=0,
            status="manual",
            message=message,
        )
        self.record_application(job_pk, rec)

    def list_jobs(
        self,
        *,
        portal: Optional[str] = None,
        applied: Optional[bool] = None,
        min_score: int = 0,
        search: str = "",
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = ["j.match_score >= ?"]
        params: list[Any] = [min_score]

        if portal:
            clauses.append("j.portal = ?")
            params.append(portal)
        if search:
            clauses.append("(j.title LIKE ? OR j.company LIKE ? OR j.description LIKE ?)")
            q = f"%{search}%"
            params.extend([q, q, q])

        applied_join = "LEFT JOIN applications a ON a.job_pk = j.id"
        if applied is True:
            clauses.append("a.job_pk IS NOT NULL AND a.status IN ('submitted','assisted','manual')")
        elif applied is False:
            clauses.append("(a.job_pk IS NULL OR a.status NOT IN ('submitted','assisted','manual'))")

        where = " AND ".join(clauses)
        params.extend([limit, offset])

        rows = self.conn.execute(
            f"""
            SELECT j.*, a.status AS apply_status, a.message AS apply_message, a.applied_at
            FROM jobs j
            {applied_join}
            WHERE {where}
            ORDER BY j.match_score DESC, j.updated_at DESC
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
        return [self._row_to_job_dict(r) for r in rows]

    def get_job(self, job_pk: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            """
            SELECT j.*, a.status AS apply_status, a.message AS apply_message, a.applied_at
            FROM jobs j
            LEFT JOIN applications a ON a.job_pk = j.id
            WHERE j.id = ?
            """,
            (job_pk,),
        ).fetchone()
        return self._row_to_job_dict(row) if row else None

    def stats(self) -> dict[str, int]:
        total = self.conn.execute("SELECT COUNT(*) AS c FROM jobs").fetchone()["c"]
        applied = self.conn.execute(
            """
            SELECT COUNT(*) AS c FROM applications
            WHERE status IN ('submitted','assisted','manual')
            """
        ).fetchone()["c"]
        recommended = self.conn.execute(
            "SELECT COUNT(*) AS c FROM jobs WHERE recommended = 1"
        ).fetchone()["c"]
        with_desc = self.conn.execute(
            "SELECT COUNT(*) AS c FROM jobs WHERE length(description) > 50"
        ).fetchone()["c"]
        return {
            "total": total,
            "applied": applied,
            "recommended": recommended,
            "with_description": with_desc,
        }

    def import_matches_json(self, path: Path) -> int:
        if not path.exists():
            return 0
        data = json.loads(path.read_text(encoding="utf-8"))
        count = 0
        for item in data:
            job_data = item.get("job", item)
            job = JobListing.model_validate(job_data)
            match = MatchResult(
                job=job,
                score=int(item.get("score", 0)),
                reasoning=str(item.get("reasoning", "")),
                skill_overlap=item.get("skill_overlap", []),
                gaps=item.get("gaps", []),
                recommended=bool(item.get("recommended", False)),
            )
            self.upsert_match(match)
            count += 1
        return count

    def match_result_for_job(self, job_pk: str) -> Optional[MatchResult]:
        row = self.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_pk,)).fetchone()
        if not row:
            return None
        job = JobListing(
            portal=Portal(row["portal"]),
            job_id=row["job_id"],
            title=row["title"],
            company=row["company"],
            location=row["location"] or "",
            url=row["url"],
            description=row["description"] or "",
            posted=row["posted"],
        )
        return MatchResult(
            job=job,
            score=row["match_score"] or 0,
            reasoning=row["reasoning"] or "",
            skill_overlap=json.loads(row["skill_overlap"] or "[]"),
            gaps=json.loads(row["gaps"] or "[]"),
            recommended=bool(row["recommended"]),
        )

    def _row_to_job_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["skill_overlap"] = json.loads(d.get("skill_overlap") or "[]")
        d["gaps"] = json.loads(d.get("gaps") or "[]")
        d["recommended"] = bool(d.get("recommended"))
        d["is_applied"] = d.get("apply_status") in APPLIED_STATUSES
        d["has_description"] = len(d.get("description") or "") > 50
        return d
