from __future__ import annotations

import asyncio
from typing import Any

from playwright.async_api import Browser, async_playwright

from src.models import JobListing, Portal

from .descriptions import fetch_all_descriptions
from .indeed import scrape_indeed
from .linkedin import scrape_linkedin
from .naukri import scrape_naukri

SCRAPERS = {
    Portal.LINKEDIN: scrape_linkedin,
    Portal.NAUKRI: scrape_naukri,
    Portal.INDEED: scrape_indeed,
}


async def scrape_all_portals(
    query: str,
    location: str,
    portals: list[str],
    max_per_portal: int,
    *,
    headless: bool = True,
    slow_mo_ms: int = 100,
    fetch_descriptions: bool = True,
) -> list[JobListing]:
    enabled = []
    for name in portals:
        try:
            enabled.append(Portal(name.lower()))
        except ValueError:
            continue

    jobs: list[JobListing] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=headless,
            slow_mo=slow_mo_ms,
        )
        try:
            for portal in enabled:
                scraper = SCRAPERS.get(portal)
                if not scraper:
                    continue
                batch = await scraper(browser, query, location, max_per_portal)
                jobs.extend(batch)
            if fetch_descriptions and jobs:
                await fetch_all_descriptions(browser, jobs)
        finally:
            await browser.close()
    return jobs


def scrape_jobs_sync(
    query: str,
    location: str,
    portals: list[str],
    max_per_portal: int,
    **kwargs: Any,
) -> list[JobListing]:
    return asyncio.run(
        scrape_all_portals(query, location, portals, max_per_portal, **kwargs)
    )
