#!/usr/bin/env python3
"""Check NeurIPS submission-format constraints for the generated paper PDF."""
from __future__ import annotations

from pathlib import Path
import re
import sys

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper/main.pdf"
MAIN = ROOT / "paper/main.tex"
MAX_MAIN_CONTENT_PAGES = 9
MAX_PDF_BYTES = 50 * 1024 * 1024


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def first_page_containing(pages: list[str], needle: str) -> int | None:
    for i, text in enumerate(pages, start=1):
        if needle in text:
            return i
    return None


def main() -> None:
    if not PDF.exists():
        fail(f"missing PDF: {PDF}")
    size = PDF.stat().st_size
    if size > MAX_PDF_BYTES:
        fail(f"PDF is {size} bytes, exceeds 50MB")

    reader = PdfReader(str(PDF))
    pages = [page.extract_text() or "" for page in reader.pages]
    refs_page = first_page_containing(pages, "References")
    checklist_page = first_page_containing(pages, "NeurIPS Paper Checklist")
    appendix_page = min(
        p for p in [
            first_page_containing(pages, "Synthetic Simulation Details"),
            first_page_containing(pages, "Per-Scenario Synthetic Results"),
        ] if p is not None
    )
    if refs_page is None:
        fail("References section not found in PDF")
    if appendix_page is None:
        fail("Appendix section not found in PDF")
    if checklist_page is None:
        fail("NeurIPS checklist not found in PDF")
    if refs_page > MAX_MAIN_CONTENT_PAGES + 1:
        fail(f"references start on page {refs_page}; main content exceeds {MAX_MAIN_CONTENT_PAGES} pages")
    if not (refs_page < appendix_page < checklist_page):
        fail(f"incorrect order: refs={refs_page}, appendix={appendix_page}, checklist={checklist_page}")

    pdf_text = "\n".join(pages)
    identity_patterns = [
        r"/mnt/",
        r"Acknowledgments and Disclosure of Funding",
        r"\bour previous\b",
        r"\bwe previously\b",
        r"\bour prior\b",
    ]
    hits = [pat for pat in identity_patterns if re.search(pat, pdf_text, re.IGNORECASE)]
    if hits:
        fail("potential deanonymization/acknowledgment text in PDF: " + ", ".join(hits))

    tex = MAIN.read_text(errors="ignore")
    if "\\usepackage[main]{neurips_2026}" not in tex:
        fail("paper does not use the NeurIPS 2026 main submission style option")
    prohibited_style_options = ["final", "preprint", "nonanonymous"]
    style_line = next((line for line in tex.splitlines() if "neurips_2026" in line and "usepackage" in line), "")
    bad_options = [opt for opt in prohibited_style_options if opt in style_line]
    if bad_options:
        fail("non-anonymous/preprint/final style option present: " + ", ".join(bad_options))

    print("NeurIPS submission checks passed")
    print(f"PDF: {PDF.relative_to(ROOT)} ({size / (1024 * 1024):.2f} MB)")
    print(f"Total pages: {len(pages)}")
    print(f"Main content pages: {refs_page - 1}")
    print(f"References start page: {refs_page}")
    print(f"Appendix start page: {appendix_page}")
    print(f"Checklist start page: {checklist_page}")


if __name__ == "__main__":
    main()
