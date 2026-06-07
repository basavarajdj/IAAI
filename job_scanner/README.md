# Job Scanner Agent

Scrape jobs from **LinkedIn**, **Naukri**, and **Indeed**, match them to your resume using local **Ollama** (`gemma4:e2b`), and optionally apply (dry-run by default).

## Prerequisites

1. [Ollama](https://ollama.com) running with your model:
   ```bash
   ollama pull gemma4:e2b
   ollama serve
   ```
2. Python 3.11+ (3.14 tested)
3. Place your resume PDF at:
   `files/Resume_Basavaraj_Jakkannavar_v1.pdf`
   (or set `RESUME_PATH` in `.env`)
   Scanned/image PDFs are read via **Ollama vision** (`gemma4:e2b`) when text extraction fails.

## Setup

```powershell
cd d:\Git\IAAI\job_scanner
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
playwright install chromium
copy .env.example .env
```

Edit `config.yaml` for default **roles** (ML engineer, data science, cybersecurity, etc.) and **industries** (healthcare, banking, manufacturing, etc.).

## Usage

```powershell
# Set interests via CLI or config.yaml
python main.py set-preferences "ML Engineer,Data Science,Cybersecurity" "Healthcare,Banking,Technology"

# Parse resume only (Ollama)
python main.py parse-resume

# Scan + match (saves to output/jobs.db + output/matches.json)
python main.py scan --roles "Data Scientist,ML Engineer" --industries "Healthcare,Banking" -l India -n 10

# Web UI — browse jobs, descriptions, applied status
python main.py ui
# Open http://127.0.0.1:8080

# Scan and run apply step (dry_run unless APPLY_MODE changed)
python main.py scan --apply

# Visible browser (login / CAPTCHA)
python main.py scan --no-headless -n 5
```

### Apply modes (`.env`)

| `APPLY_MODE` | Behavior |
|--------------|----------|
| `dry_run` | Log what would be applied (default) |
| `assisted` | Open browser, start Easy Apply; you confirm submit |
| `auto` | Best-effort submit (LinkedIn Easy Apply / Naukri); may fail on custom forms |

Set portal credentials in `.env` for logged-in apply:

```
LINKEDIN_EMAIL=...
LINKEDIN_PASSWORD=...
NAUKRI_EMAIL=...
NAUKRI_PASSWORD=...
```

## Output

- `output/jobs.db` — SQLite store (jobs, descriptions, application tracking)
- `output/matches.json` — ranked jobs with scores and reasoning (JSON export)
- `output/applications.jsonl` — apply audit log

## Web UI

```powershell
python main.py ui --port 8080
```

Features:
- List all scraped jobs with match scores and filters (portal, applied/not applied, min score)
- View full job descriptions and match reasoning
- **Apply** button (skips if already applied)
- **Mark as applied** for jobs you applied to manually

## Important notes

- Job sites change HTML often; scrapers may need selector updates.
- Auto-apply may violate site terms; use `assisted` mode and review each application.
- Indeed auto-apply is not implemented; URLs are matched and listed for manual apply.
- All LLM calls use **local Ollama only** — no cloud API keys.

## Architecture

```
main.py (CLI)
  → resume/parser.py     PDF + Ollama profile
  → scrapers/            Playwright: LinkedIn, Naukri, Indeed
  → matching/matcher.py  Ollama fit score 0–100
  → apply/applier.py     Playwright apply + Ollama form answers
```
