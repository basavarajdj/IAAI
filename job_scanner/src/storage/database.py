from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    portal TEXT NOT NULL,
    job_id TEXT NOT NULL,
    title TEXT NOT NULL,
    company TEXT NOT NULL,
    location TEXT DEFAULT '',
    url TEXT NOT NULL UNIQUE,
    description TEXT DEFAULT '',
    posted TEXT,
    scraped_at TEXT,
    match_score INTEGER DEFAULT 0,
    reasoning TEXT DEFAULT '',
    skill_overlap TEXT DEFAULT '[]',
    gaps TEXT DEFAULT '[]',
    recommended INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_pk TEXT NOT NULL UNIQUE REFERENCES jobs(id),
    status TEXT NOT NULL,
    message TEXT DEFAULT '',
    applied_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_score ON jobs(match_score DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_portal ON jobs(portal);
CREATE INDEX IF NOT EXISTS idx_applications_status ON applications(status);
"""


def get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn
