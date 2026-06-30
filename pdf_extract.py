# -*- coding: utf-8 -*-
"""Lightweight PDF extraction for manuscript PDFs (title / abstract / reference count / type).

Tuned for ScholarOne 'For Review Only' exports but falls back gracefully for generic PDFs.
"""
import re
import fitz  # PyMuPDF


def count_references(clean):
    refm = (list(re.finditer(r"(?im)^\s*references?\s*$", clean))
            or list(re.finditer(r"(?i)\breferences\b", clean)))
    if not refm:
        return 0, False
    reftext = clean[refm[-1].start():]
    cands = [
        len(re.findall(r"(?m)^\s*\[\d{1,3}\]", reftext)),
        len(re.findall(r"(?m)^\s*\d{1,3}\.\s+\D", reftext)),
        len(re.findall(r"(?m)^\s*\d{1,3}\.\s*$", reftext)),
        len(re.findall(r"(?m)^\s*\d{1,3}\)\s", reftext)),
        len(re.findall(r"(?m)^\s*\d{1,3}\s{2,}\D", reftext)),
    ]
    best = max(cands)
    if best >= 5:
        return best, True
    yr = len(re.findall(r"\b(?:19|20)\d{2}\b", re.sub(r"\s+", " ", reftext)))
    if yr >= 5:
        return yr, False
    return best, False


def extract_pdf(path):
    doc = fitz.open(path)
    p1 = doc[0].get_text()
    full = "\n".join(doc[i].get_text() for i in range(doc.page_count))
    lines = [l for l in full.split("\n") if not re.fullmatch(r"\s*\d{1,3}\s*", l)]
    clean = "\n".join(lines)

    mid = re.search(r"ASJ-(\d{4})-(\d{4})", p1) or re.search(r"ASJ-(\d{4})-(\d{4})", clean)
    msid = mid.group(0) if mid else None

    title = ""
    mt = re.search(r"For Review Only\s*(.*?)\s*Journal:", p1, re.S)
    if mt:
        title = re.sub(r"\s+", " ", mt.group(1)).strip()
    if not title:
        for l in p1.split("\n"):
            if len(l.strip()) > 20 and "Review Only" not in l:
                title = l.strip()
                break

    is_review = 0
    mtype = re.search(r"Manuscript Type:\s*(.+)", p1)
    if mtype and "review" in mtype.group(1).lower():
        is_review = 1

    abstract = ""
    ma = re.search(r"(?i)\babstract\b", clean)
    if ma:
        tail = clean[ma.end():]
        end = re.search(r"(?i)\n\s*keywords\b|\n\s*introduction\s*\n|\n\s*references\b", tail)
        abstract = tail[:end.start()] if end else tail[:2500]
        abstract = re.sub(r"\s+", " ", abstract).strip()

    n_refs, ref_conf = count_references(clean)
    doc.close()
    low_conf = (not ref_conf and n_refs < 5) or len(abstract.split()) < 30
    return {
        "msid": msid, "title": title, "abstract": abstract,
        "is_review": is_review, "n_references": n_refs, "low_conf": low_conf,
    }
