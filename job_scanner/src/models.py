from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Portal(str, Enum):
    LINKEDIN = "linkedin"
    NAUKRI = "naukri"
    INDEED = "indeed"


class JobListing(BaseModel):
    portal: Portal
    job_id: str
    title: str
    company: str
    location: str = ""
    url: str
    description: str = ""
    posted: Optional[str] = None
    scraped_at: datetime = Field(default_factory=datetime.utcnow)


class ResumeProfile(BaseModel):
    raw_text: str
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    experience_years: Optional[float] = None
    preferred_titles: list[str] = Field(default_factory=list)


class MatchResult(BaseModel):
    job: JobListing
    score: int = Field(ge=0, le=100)
    reasoning: str = ""
    skill_overlap: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    recommended: bool = False


class ApplicationRecord(BaseModel):
    job_url: str
    portal: Portal
    title: str
    company: str
    match_score: int
    status: str  # dry_run | submitted | failed | skipped
    message: str = ""
    applied_at: datetime = Field(default_factory=datetime.utcnow)
