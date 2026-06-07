#!/usr/bin/env python3
"""Scan job portals, match resume via Ollama, optionally apply."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from src.apply.applier import JobApplier
from src.llm.ollama_client import OllamaClient
from src.matching.matcher import JobMatcher
from src.resume.parser import load_resume_profile
from src.scrapers.base import scrape_jobs_sync
from src.settings import EnvSettings, load_yaml_config, resolve_resume_path
from src.storage.job_store import JobStore

app = typer.Typer(help="Job scanner: LinkedIn, Naukri, Indeed + Ollama matching")
console = Console()


def _build_query(roles: list[str]) -> str:
    return " OR ".join(roles[:3]) if roles else "software engineer"


@app.command()
def scan(
    roles: Optional[str] = typer.Option(
        None,
        "--roles",
        help="Comma-separated interests e.g. 'ML Engineer,Data Science,Cybersecurity'",
    ),
    industries: Optional[str] = typer.Option(
        None,
        "--industries",
        help="Comma-separated e.g. 'Healthcare,Banking,Manufacturing'",
    ),
    location: str = typer.Option("India", "--location", "-l"),
    max_jobs: int = typer.Option(15, "--max", "-n"),
    apply: bool = typer.Option(False, "--apply", help="Run apply step (respects APPLY_MODE)"),
    headless: Optional[bool] = typer.Option(None, "--headless/--no-headless"),
):
    """Scrape jobs, match against resume, show ranked results; optionally apply."""
    env = EnvSettings()
    yaml_cfg = load_yaml_config()
    prefs = yaml_cfg.get("preferences", {})
    search = yaml_cfg.get("search", {})
    output_cfg = yaml_cfg.get("output", {})

    if roles:
        prefs["roles"] = [r.strip() for r in roles.split(",") if r.strip()]
    if industries:
        prefs["industries"] = [i.strip() for i in industries.split(",") if i.strip()]

    query = _build_query(prefs.get("roles", []))
    portals = search.get("portals", ["linkedin", "naukri", "indeed"])
    max_per = min(max_jobs, search.get("max_jobs_per_portal", 25))

    resume_path = resolve_resume_path(env)
    llm = OllamaClient(env.ollama_host, env.ollama_model)

    console.print(f"[bold]Model:[/bold] {env.ollama_model} @ {env.ollama_host}")
    console.print(f"[bold]Resume:[/bold] {resume_path}")
    console.print(f"[bold]Query:[/bold] {query} | [bold]Location:[/bold] {location}")
    console.print(f"[bold]Roles:[/bold] {', '.join(prefs.get('roles', []))}")
    console.print(f"[bold]Industries:[/bold] {', '.join(prefs.get('industries', []))}")

    console.print("\n[cyan]Parsing resume with Ollama...[/cyan]")
    profile = load_resume_profile(resume_path, llm)

    use_headless = env.playwright_headless if headless is None else headless
    console.print("\n[cyan]Scraping job portals (Playwright)...[/cyan]")
    jobs = scrape_jobs_sync(
        query,
        location,
        portals,
        max_per,
        headless=use_headless,
        slow_mo_ms=env.playwright_slow_mo_ms,
    )
    console.print(f"Found [green]{len(jobs)}[/green] listings")

    if not jobs:
        raise typer.Exit(1)

    console.print("\n[cyan]Matching jobs with Ollama...[/cyan]")
    matcher = JobMatcher(llm, min_score=env.min_match_score)
    ranked = matcher.rank_jobs(jobs, profile, prefs)

    results_dir = Path(output_cfg.get("results_dir", "output"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "matches.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(
            [r.model_dump(mode="json") for r in ranked],
            f,
            indent=2,
            default=str,
        )
    console.print(f"Saved matches to [green]{out_file}[/green]")

    db_path = results_dir / "jobs.db"
    store = JobStore(db_path)
    try:
        pks = store.save_matches(ranked)
        console.print(f"Stored [green]{len(pks)}[/green] jobs in local DB [green]{db_path}[/green]")
    finally:
        store.close()

    table = Table(title="Top matches")
    table.add_column("Score", justify="right")
    table.add_column("Portal")
    table.add_column("Title")
    table.add_column("Company")
    table.add_column("Rec?")

    for r in ranked[:15]:
        table.add_row(
            str(r.score),
            r.job.portal.value,
            r.job.title[:40],
            r.job.company[:25],
            "✓" if r.recommended else "",
        )
    console.print(table)

    top = [r for r in ranked if r.recommended][:5]
    if top:
        console.print("\n[bold]Best fits:[/bold]")
        for r in top:
            console.print(f"  [{r.score}] {r.job.title} @ {r.job.company}")
            console.print(f"      {r.job.url}")
            if r.reasoning:
                console.print(f"      {r.reasoning[:120]}...")

    if apply:
        log_path = Path(output_cfg.get("applications_log", "output/applications.jsonl"))
        store = JobStore(db_path)
        try:
            applier = JobApplier(llm, env, log_path, store=store)
            console.print(f"\n[cyan]Apply mode: {env.apply_mode}[/cyan]")
            records = applier.apply_matches(top, profile)
            for rec in records:
                console.print(f"  [{rec.status}] {rec.title} — {rec.message}")
        finally:
            store.close()


@app.command()
def parse_resume():
    """Only parse resume and print profile summary (Ollama)."""
    env = EnvSettings()
    llm = OllamaClient(env.ollama_host, env.ollama_model)
    path = resolve_resume_path(env)
    profile = load_resume_profile(path, llm)
    console.print_json(profile.model_dump_json(indent=2))


@app.command()
def set_preferences(
    roles: str = typer.Argument(..., help="Comma-separated job interests"),
    industries: str = typer.Argument("", help="Comma-separated industries"),
):
    """Update config.yaml roles and industries."""
    import yaml

    cfg_path = Path(__file__).parent / "config.yaml"
    data = load_yaml_config(cfg_path)
    data.setdefault("preferences", {})
    data["preferences"]["roles"] = [r.strip() for r in roles.split(",") if r.strip()]
    if industries:
        data["preferences"]["industries"] = [
            i.strip() for i in industries.split(",") if i.strip()
        ]
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    console.print("[green]Updated config.yaml preferences[/green]")


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8080, "--port"),
    reload: bool = typer.Option(False, "--reload"),
):
    """Launch web UI to browse jobs, descriptions, and track applications."""
    import uvicorn

    from src.ui.app import create_app

    yaml_cfg = load_yaml_config()
    output_cfg = yaml_cfg.get("output", {})
    db_path = Path(output_cfg.get("results_dir", "output")) / "jobs.db"
    fastapi_app = create_app(db_path)

    console.print(f"[green]Job Scanner UI[/green] at http://{host}:{port}")
    uvicorn.run(fastapi_app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
