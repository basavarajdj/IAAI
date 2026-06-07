from __future__ import annotations

import tempfile
from pathlib import Path

import fitz
from pypdf import PdfReader

from src.llm.ollama_client import OllamaClient
from src.models import ResumeProfile

VISION_EXTRACT_PROMPT = """Extract ALL text from these resume page images.
Return plain text only, preserving sections (experience, education, skills)."""


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def extract_pdf_text_vision(path: Path, llm: OllamaClient) -> str:
    """OCR via Ollama vision when PDF has no embedded text."""
    doc = fitz.open(str(path))
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        image_paths: list[Path] = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=150)
            img_path = tmp_path / f"page_{i}.png"
            pix.save(str(img_path))
            image_paths.append(img_path)
        doc.close()
        if not image_paths:
            return ""
        return llm.generate_from_images(VISION_EXTRACT_PROMPT, image_paths)


RESUME_SYSTEM = """You analyze resumes for job matching. Respond with valid JSON only, no markdown.
Schema:
{
  "summary": "2-4 sentence professional summary",
  "skills": ["skill1", "skill2"],
  "experience_years": number or null,
  "preferred_titles": ["title1", "title2"]
}"""


def load_resume_profile(
    path: Path,
    llm: OllamaClient,
    *,
    use_llm: bool = True,
) -> ResumeProfile:
    if not path.exists():
        raise FileNotFoundError(
            f"Resume not found at {path}. Place your PDF in files/ or set RESUME_PATH in .env"
        )

    raw = extract_pdf_text(path)
    if not raw:
        raw = extract_pdf_text_vision(path, llm)
    if not raw:
        raise ValueError(f"Could not extract text from {path}")

    if not use_llm:
        return ResumeProfile(raw_text=raw)

    prompt = f"""Parse this resume text into the JSON schema.

RESUME:
{raw[:12000]}
"""
    data = llm.generate_json(prompt, system=RESUME_SYSTEM)
    return ResumeProfile(
        raw_text=raw,
        summary=str(data.get("summary", "")),
        skills=[str(s) for s in data.get("skills", [])],
        experience_years=data.get("experience_years"),
        preferred_titles=[str(t) for t in data.get("preferred_titles", [])],
    )
