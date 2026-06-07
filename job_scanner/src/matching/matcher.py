from __future__ import annotations

from typing import Any

from src.llm.ollama_client import OllamaClient
from src.models import JobListing, MatchResult, ResumeProfile

MATCH_SYSTEM = """You score job fit against a candidate resume and stated interests.
Respond with valid JSON only:
{
  "score": 0-100 integer,
  "reasoning": "brief explanation",
  "skill_overlap": ["matched skills"],
  "gaps": ["missing requirements"],
  "recommended": true if score >= threshold else false
}"""


class JobMatcher:
    def __init__(self, llm: OllamaClient, min_score: int = 70) -> None:
        self.llm = llm
        self.min_score = min_score

    def match_job(
        self,
        job: JobListing,
        profile: ResumeProfile,
        preferences: dict[str, Any],
    ) -> MatchResult:
        roles = preferences.get("roles", [])
        industries = preferences.get("industries", [])
        keywords = preferences.get("keywords", [])
        exclude = preferences.get("exclude_keywords", [])

        title_lower = job.title.lower()
        desc_lower = (job.description or job.title).lower()
        for ex in exclude:
            if ex.lower() in title_lower or ex.lower() in desc_lower:
                return MatchResult(
                    job=job,
                    score=0,
                    reasoning=f"Excluded keyword: {ex}",
                    recommended=False,
                )

        prompt = f"""Score this job for the candidate.

TARGET ROLES: {", ".join(roles)}
TARGET INDUSTRIES: {", ".join(industries)}
PREFERRED KEYWORDS: {", ".join(keywords)}
MINIMUM RECOMMENDED SCORE: {self.min_score}

CANDIDATE SUMMARY:
{profile.summary or profile.raw_text[:2000]}

CANDIDATE SKILLS: {", ".join(profile.skills[:40])}
EXPERIENCE (years): {profile.experience_years}

JOB:
Title: {job.title}
Company: {job.company}
Location: {job.location}
Portal: {job.portal.value}
Description:
{(job.description or "No description scraped")[:6000]}
"""
        data = self.llm.generate_json(prompt, system=MATCH_SYSTEM)
        score = int(data.get("score", 0))
        score = max(0, min(100, score))
        recommended = bool(data.get("recommended", score >= self.min_score))
        if score >= self.min_score:
            recommended = True

        return MatchResult(
            job=job,
            score=score,
            reasoning=str(data.get("reasoning", "")),
            skill_overlap=[str(s) for s in data.get("skill_overlap", [])],
            gaps=[str(g) for g in data.get("gaps", [])],
            recommended=recommended,
        )

    def rank_jobs(
        self,
        jobs: list[JobListing],
        profile: ResumeProfile,
        preferences: dict[str, Any],
    ) -> list[MatchResult]:
        results: list[MatchResult] = []
        for job in jobs:
            results.append(self.match_job(job, profile, preferences))
        results.sort(key=lambda r: r.score, reverse=True)
        return results
