from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

from playwright.async_api import Browser, Page, async_playwright

from src.llm.ollama_client import OllamaClient
from src.models import ApplicationRecord, MatchResult, Portal, ResumeProfile
from src.storage.job_store import JobStore
from src.storage.keys import normalize_job_key

ANSWER_SYSTEM = """Generate short, honest application answers from the resume only.
Respond with valid JSON: {"answer": "text under 500 chars"}"""


class JobApplier:
    def __init__(
        self,
        llm: OllamaClient,
        env: Any,
        log_path: Path,
        store: Optional[JobStore] = None,
    ) -> None:
        self.llm = llm
        self.env = env
        self.log_path = log_path
        self.store = store
        self.mode = env.apply_mode.lower()

    def apply_matches(self, matches: list[MatchResult], profile: ResumeProfile) -> list[ApplicationRecord]:
        return asyncio.run(self._apply_async(matches, profile))

    def apply_single(self, match: MatchResult, profile: ResumeProfile) -> ApplicationRecord:
        return asyncio.run(self._apply_async([match], profile))[0]

    async def _apply_async(
        self,
        matches: list[MatchResult],
        profile: ResumeProfile,
    ) -> list[ApplicationRecord]:
        records: list[ApplicationRecord] = []
        to_apply: list[MatchResult] = []

        for m in matches:
            if not m.recommended and self.mode != "dry_run":
                continue
            job_pk = normalize_job_key(m.job.portal, m.job.job_id, m.job.url)
            if self.store and self.store.is_applied(job_pk):
                records.append(
                    ApplicationRecord(
                        job_url=m.job.url,
                        portal=m.job.portal,
                        title=m.job.title,
                        company=m.job.company,
                        match_score=m.score,
                        status="skipped",
                        message="Already applied — skipped duplicate.",
                    )
                )
                continue
            to_apply.append(m)

        if self.mode == "dry_run":
            for m in to_apply:
                rec = ApplicationRecord(
                    job_url=m.job.url,
                    portal=m.job.portal,
                    title=m.job.title,
                    company=m.job.company,
                    match_score=m.score,
                    status="dry_run",
                    message="Would apply (dry_run mode). Set APPLY_MODE=assisted or auto.",
                )
                records.append(rec)
                self._persist(rec, m)
            return records

        headless = self.env.playwright_headless
        slow_mo = self.env.playwright_slow_mo_ms

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless, slow_mo=slow_mo)
            try:
                for m in to_apply:
                    job_pk = normalize_job_key(m.job.portal, m.job.job_id, m.job.url)
                    if self.store and self.store.is_applied(job_pk):
                        rec = ApplicationRecord(
                            job_url=m.job.url,
                            portal=m.job.portal,
                            title=m.job.title,
                            company=m.job.company,
                            match_score=m.score,
                            status="skipped",
                            message="Already applied — skipped duplicate.",
                        )
                        records.append(rec)
                        continue
                    rec = await self._apply_one(browser, m, profile)
                    records.append(rec)
                    self._persist(rec, m)
            finally:
                await browser.close()
        return records

    def _persist(self, record: ApplicationRecord, match: MatchResult) -> None:
        self._append_log(record)
        if self.store and record.status not in ("skipped", "failed", "dry_run"):
            job_pk = normalize_job_key(match.job.portal, match.job.job_id, match.job.url)
            self.store.record_application(job_pk, record)

    async def _apply_one(
        self,
        browser: Browser,
        match: MatchResult,
        profile: ResumeProfile,
    ) -> ApplicationRecord:
        job = match.job
        page = await browser.new_page()
        try:
            if job.portal == Portal.LINKEDIN:
                await self._ensure_linkedin_login(page)
                status, msg = await self._apply_linkedin(page, job.url, profile)
            elif job.portal == Portal.NAUKRI:
                await self._ensure_naukri_login(page)
                status, msg = await self._apply_naukri(page, job.url, profile)
            else:
                status, msg = "skipped", f"No auto-apply for {job.portal.value}; open URL manually."
                if self.mode == "assisted":
                    await page.goto(job.url, wait_until="domcontentloaded")
                    msg = "Browser opened for manual apply (assisted mode)."

            return ApplicationRecord(
                job_url=job.url,
                portal=job.portal,
                title=job.title,
                company=job.company,
                match_score=match.score,
                status=status,
                message=msg,
            )
        except Exception as e:
            return ApplicationRecord(
                job_url=job.url,
                portal=job.portal,
                title=job.title,
                company=job.company,
                match_score=match.score,
                status="failed",
                message=str(e),
            )
        finally:
            if self.mode != "assisted":
                await page.close()

    async def _ensure_linkedin_login(self, page: Page) -> None:
        email, password = self.env.linkedin_email, self.env.linkedin_password
        if not email or not password:
            return
        await page.goto("https://www.linkedin.com/login", wait_until="domcontentloaded")
        if await page.locator("input#username").count():
            await page.fill("input#username", email)
            await page.fill("input#password", password)
            await page.click("button[type=submit]")
            await page.wait_for_timeout(5000)

    async def _ensure_naukri_login(self, page: Page) -> None:
        email, password = self.env.naukri_email, self.env.naukri_password
        if not email or not password:
            return
        await page.goto("https://www.naukri.com/nlogin/login", wait_until="domcontentloaded")
        if await page.locator("input#usernameField").count():
            await page.fill("input#usernameField", email)
            await page.fill("input#passwordField", password)
            await page.click("button[type=submit]")
            await page.wait_for_timeout(5000)

    async def _apply_linkedin(self, page: Page, url: str, profile: ResumeProfile) -> tuple[str, str]:
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(2000)
        easy = page.locator("button.jobs-apply-button, button[aria-label*='Easy Apply']")
        if not await easy.count():
            return "skipped", "No Easy Apply button; apply manually on LinkedIn."
        await easy.first.click()
        await page.wait_for_timeout(2000)

        if self.mode == "assisted":
            return "assisted", "Easy Apply started — complete remaining steps in browser."

        for _ in range(8):
            submit = page.locator("button[aria-label='Submit application'], button:has-text('Submit application')")
            if await submit.count():
                if self.mode == "auto":
                    await submit.first.click()
                    return "submitted", "LinkedIn Easy Apply submitted."
                return "assisted", "Review and click Submit in browser."

            await self._fill_visible_questions(page, profile)

            nxt = page.locator("button[aria-label='Continue to next step'], button:has-text('Next')")
            if await nxt.count():
                await nxt.first.click()
                await page.wait_for_timeout(1500)
            else:
                break
        return "assisted", "Partial Easy Apply — finish in browser."

    async def _apply_naukri(self, page: Page, url: str, profile: ResumeProfile) -> tuple[str, str]:
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(2000)
        apply_btn = page.locator("button#apply-button, button.apply-button, a#apply-button")
        if not await apply_btn.count():
            return "skipped", "No Apply button found on Naukri listing."
        await apply_btn.first.click()
        await page.wait_for_timeout(2000)
        await self._fill_visible_questions(page, profile)

        if self.mode == "auto":
            send = page.locator("button:has-text('Apply'), button:has-text('Send')")
            if await send.count():
                await send.first.click()
                return "submitted", "Naukri apply flow completed (verify in account)."
        return "assisted", "Naukri apply opened — confirm submission in browser."

    async def _fill_visible_questions(self, page: Page, profile: ResumeProfile) -> None:
        inputs = page.locator("textarea:visible, input[type=text]:visible")
        count = await inputs.count()
        for i in range(min(count, 5)):
            el = inputs.nth(i)
            label = await el.get_attribute("aria-label") or await el.get_attribute("name") or "question"
            prompt = f"""Question/field: {label}
Resume excerpt: {profile.raw_text[:3000]}
Provide a truthful short answer."""
            try:
                data = self.llm.generate_json(prompt, system=ANSWER_SYSTEM)
                answer = str(data.get("answer", ""))[:500]
                if answer:
                    await el.fill(answer)
            except Exception:
                pass

    def _append_log(self, record: ApplicationRecord) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
