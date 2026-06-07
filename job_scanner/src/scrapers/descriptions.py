from __future__ import annotations

from playwright.async_api import Browser, Page

from src.models import JobListing, Portal


async def fetch_all_descriptions(browser: Browser, jobs: list[JobListing]) -> None:
    """Download full descriptions for every job listing."""
    by_portal: dict[Portal, list[JobListing]] = {}
    for job in jobs:
        by_portal.setdefault(job.portal, []).append(job)

    for portal, batch in by_portal.items():
        page = await browser.new_page()
        try:
            for job in batch:
                if len(job.description) > 50:
                    continue
                desc = await _fetch(page, portal, job.url)
                if desc:
                    job.description = desc
        finally:
            await page.close()


async def _fetch(page: Page, portal: Portal, job_url: str) -> str:
    try:
        await page.goto(job_url, wait_until="domcontentloaded", timeout=45000)
        await page.wait_for_timeout(1500)

        if portal == Portal.LINKEDIN:
            sel = "div.show-more-less-html__markup, div.description__text, article.jobs-description__container"
        elif portal == Portal.NAUKRI:
            sel = "div.dang-inner-html, section.job-desc"
        else:
            sel = "#jobDescriptionText, div.jobsearch-jobDescriptionText"

        desc = page.locator(sel)
        if await desc.count():
            return (await desc.first.inner_text()).strip()[:12000]
    except Exception:
        pass
    return ""
