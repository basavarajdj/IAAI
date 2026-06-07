from __future__ import annotations

import re

from src.models import Portal


def normalize_job_key(portal: Portal | str, job_id: str, url: str) -> str:
    """Stable primary key: portal + numeric/slug id from URL."""
    portal_val = portal.value if isinstance(portal, Portal) else str(portal)
    clean = _extract_stable_id(portal_val, job_id, url)
    return f"{portal_val}:{clean}"


def _extract_stable_id(portal: str, job_id: str, url: str) -> str:
    url = url.split("?")[0].rstrip("/")

    if portal == "linkedin":
        m = re.search(r"(\d{6,})$", url) or re.search(r"currentJobId=(\d+)", job_id)
        if m:
            return m.group(1)

    if portal == "naukri":
        m = re.search(r"job-listings-(\d+)", url) or re.search(r"(\d{6,})$", url)
        if m:
            return m.group(1)

    if portal == "indeed":
        m = re.search(r"jk=([a-f0-9]+)", url + "?" + job_id)
        if m:
            return m.group(1)

    slug = url.split("/")[-1]
    slug = re.sub(r"\?.*", "", slug)
    if slug and len(slug) < 120:
        return slug
    return re.sub(r"[^\w\-]", "_", job_id)[:120] or url
