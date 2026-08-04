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

    def _is_excluded(self, job: JobListing, preferences: dict[str, Any]) -> bool:
        exclude_keywords = preferences.get("exclude_keywords", [])
        exclude_companies = preferences.get("exclude_companies", [])

        company_lower = job.company.lower()
        for ex in exclude_companies:
            if ex.lower() in company_lower:
                return True

        title_lower = job.title.lower()
        desc_lower = (job.description or job.title).lower()
        for ex in exclude_keywords:
            if ex.lower() in title_lower or ex.lower() in desc_lower:
                return True

        return False

    def match_job(
        self,
        job: JobListing,
        profile: ResumeProfile,
        preferences: dict[str, Any],
    ) -> MatchResult:
        roles = preferences.get("roles", [])
        industries = preferences.get("industries", [])
        keywords = preferences.get("keywords", [])

        if self._is_excluded(job, preferences):
            return MatchResult(
                job=job,
                score=0,
                reasoning="Excluded by company or keyword filter",
                recommended=False,
            )

        prompt = f"""Score this job for the candidate based on the following criteria, in order of priority:

1. EXPERIENCE LEVEL — how well the candidate's experience years and seniority match the job
2. SKILLS — overlap between candidate skills and job requirements
3. JOB TITLE — relevance of the job title to the candidate's preferred roles
4. INDUSTRY — whether the job's industry matches target industries
5. JOB DESCRIPTION — overall fit based on full description

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
        score = max(0, min(100, int(data.get("score", 0))))
        recommended = score >= self.min_score or bool(data.get("recommended", False))

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
        results = [self.match_job(job, profile, preferences) for job in jobs]
        results.sort(key=lambda r: r.score, reverse=True)
        return results
