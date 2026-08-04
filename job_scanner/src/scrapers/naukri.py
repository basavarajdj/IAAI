from __future__ import annotations

import re
import urllib.parse
from datetime import datetime, timezone

from playwright.async_api import Browser

from src.models import JobListing, Portal

_POSTED_SELECTORS = [
    "span.post-day",
    "span.job-post-day",
    "div.job-post-days",
    "span[data-posted]",
]


def _parse_posted_days(text: str) -> int | None:
    text = text.strip().lower()
    m = re.search(r"(\d+)\s*days?\s*ago", text)
    if m:
        return int(m.group(1))
    if re.search(r"today", text):
        return 0
    if re.search(r"yesterday", text):
        return 1
    m = re.search(r"(\d+)\s*(hours?|hrs?)\s*ago", text)
    if m:
        return 0
    return None


async def scrape_naukri(
    browser: Browser,
    query: str,
    location: str,
    limit: int,
    *,
    max_days_old: int = 7,
) -> list[JobListing]:
    page = await browser.new_page()
    jobs: list[JobListing] = []
    try:
        q = urllib.parse.quote(query.replace(" ", "-"))
        loc = urllib.parse.quote(location.replace(" ", "-")) if location else ""
        url = (
            f"https://www.naukri.com/{q}-jobs-in-{loc}"
            if loc
            else f"https://www.naukri.com/{q}-jobs"
        )
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(3000)

        rows = page.locator("div.cust-job-tuple, article.jobTuple")
        count = await rows.count()
        seen: set[str] = set()

        for i in range(min(count, limit * 2)):
            if len(jobs) >= limit:
                break
            row = rows.nth(i)
            title_el = row.locator("a.title, h2 a")
            company_el = row.locator("a.comp-name, a.subTitle")
            loc_el = row.locator("span.locWdth, li.location")

            if not await title_el.count():
                continue
            title = (await title_el.first.inner_text()).strip()
            href = await title_el.first.get_attribute("href")
            if not href:
                continue
            if not href.startswith("http"):
                href = "https://www.naukri.com" + href
            job_id = _naukri_id(href)
            if job_id in seen:
                continue
            seen.add(job_id)
            company = (
                (await company_el.first.inner_text()).strip()
                if await company_el.count()
                else "Unknown"
            )
            loc_text = (
                (await loc_el.first.inner_text()).strip() if await loc_el.count() else ""
            )

            posted_text = ""
            for sel in _POSTED_SELECTORS:
                el = row.locator(sel)
                if await el.count():
                    posted_text = (await el.first.inner_text()).strip()
                    break
            days = _parse_posted_days(posted_text) if posted_text else None
            if days is not None and days > max_days_old:
                continue

            jobs.append(
                JobListing(
                    portal=Portal.NAUKRI,
                    job_id=job_id,
                    title=title,
                    company=company,
                    location=loc_text,
                    url=href,
                    posted=posted_text,
                )
            )
    finally:
        await page.close()
    return jobs


def _naukri_id(url: str) -> str:
    m = re.search(r"job-listings-(\d+)|/(\d+)(?:\?|$)", url)
    if m:
        return m.group(1) or m.group(2) or url
    return url
