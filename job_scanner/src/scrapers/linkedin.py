from __future__ import annotations

import re
import urllib.parse

from playwright.async_api import Browser

from src.models import JobListing, Portal


async def scrape_linkedin(
    browser: Browser,
    query: str,
    location: str,
    limit: int,
) -> list[JobListing]:
    page = await browser.new_page()
    jobs: list[JobListing] = []
    try:
        params = urllib.parse.urlencode({"keywords": query, "location": location})
        url = f"https://www.linkedin.com/jobs/search/?{params}"
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(3000)

        cards = page.locator("div.base-card, li.jobs-search-results__list-item")
        count = await cards.count()
        seen: set[str] = set()

        for i in range(min(count, limit * 2)):
            if len(jobs) >= limit:
                break
            card = cards.nth(i)
            title_el = card.locator("h3.base-search-card__title, a.job-card-list__title")
            company_el = card.locator(
                "h4.base-search-card__subtitle, a.job-card-container__company-name"
            )
            loc_el = card.locator("span.job-search-card__location")
            link_el = card.locator("a.base-card__full-link, a.job-card-list__title").first

            title = (await title_el.inner_text(timeout=2000)).strip() if await title_el.count() else ""
            if not title:
                continue
            company = (
                (await company_el.inner_text(timeout=2000)).strip()
                if await company_el.count()
                else "Unknown"
            )
            loc = (await loc_el.inner_text(timeout=2000)).strip() if await loc_el.count() else ""
            href = await link_el.get_attribute("href") if await link_el.count() else None
            if not href:
                continue
            job_id = _linkedin_id(href)
            if job_id in seen:
                continue
            seen.add(job_id)
            jobs.append(
                JobListing(
                    portal=Portal.LINKEDIN,
                    job_id=job_id,
                    title=title,
                    company=company,
                    location=loc,
                    url=href.split("?")[0],
                )
            )

    finally:
        await page.close()
    return jobs


def _linkedin_id(url: str) -> str:
    m = re.search(r"currentJobId=(\d+)|/view/(\d+)", url)
    if m:
        return m.group(1) or m.group(2) or url
    return url.rstrip("/").split("/")[-1]

