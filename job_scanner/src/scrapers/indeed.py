from __future__ import annotations

import re
import urllib.parse

from playwright.async_api import Browser

from src.models import JobListing, Portal


async def scrape_indeed(
    browser: Browser,
    query: str,
    location: str,
    limit: int,
) -> list[JobListing]:
    page = await browser.new_page()
    jobs: list[JobListing] = []
    try:
        params = urllib.parse.urlencode({"q": query, "l": location})
        url = f"https://www.indeed.com/jobs?{params}"
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(3000)

        cards = page.locator("div.job_seen_beacon, td.resultContent")
        count = await cards.count()
        seen: set[str] = set()

        for i in range(min(count, limit * 2)):
            if len(jobs) >= limit:
                break
            card = cards.nth(i)
            title_el = card.locator("h2.jobTitle a, a.jcs-JobTitle")
            company_el = card.locator("span.companyName, span[data-testid='company-name']")
            loc_el = card.locator("div.companyLocation, div[data-testid='text-location']")

            if not await title_el.count():
                continue
            title = (await title_el.first.inner_text()).strip()
            href = await title_el.first.get_attribute("href")
            if not href:
                continue
            if href.startswith("/"):
                href = "https://www.indeed.com" + href
            job_id = _indeed_id(href)
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
            jobs.append(
                JobListing(
                    portal=Portal.INDEED,
                    job_id=job_id,
                    title=title,
                    company=company,
                    location=loc_text,
                    url=href,
                )
            )

    finally:
        await page.close()
    return jobs


def _indeed_id(url: str) -> str:
    m = re.search(r"jk=([a-f0-9]+)", url)
    return m.group(1) if m else url
