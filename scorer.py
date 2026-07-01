# -*- coding: utf-8 -*-
"""Self-contained citation-impact scorer for spine manuscripts (leakage-safe final models).

Reuses the trained Model B (within-journal top 25%), Model A (field top 10%),
and the 3-year citation regressor, plus precomputed corpus constants — no full
corpus parquet is required. Text embeddings use Sentence-BERT (all-MiniLM-L6-v2).
"""
import os, re, json
from functools import lru_cache
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import normalize

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
C = json.load(open(os.path.join(MODELS, "constants.json"), encoding="utf-8"))

# study-design / signal-word detectors (same rules used in training)
RX = {
    "is_rct": r"\brandomi[sz]ed|\bRCT\b",
    "is_sr_meta": r"meta-?analysis|systematic review",
    "is_cohort": r"\bcohort\b|prospective|retrospective",
    "is_case_report": r"\bcase report|case series",
    "sw_novel": r"\bnovel\b|first[- ]time",
    "sw_ai": r"machine learning|deep learning|artificial intelligence|\bAI\b|radiomic|neural network",
    "sw_guideline": r"guideline|consensus|recommendation",
}

# UI-only heuristic (not a model feature): flags likely review/meta-analysis
# manuscripts so the "리뷰/메타분석 논문" checkbox can be pre-suggested to the
# user in the manual-entry tab, where no "Manuscript Type" field exists.
REVIEW_HINT_RX = re.compile(
    r"\bnarrative review\b|\bsystematic review\b|\bscoping review\b|\bumbrella review\b"
    r"|\bstate-of-the-art review\b|\breview article\b|\bliterature review\b|meta-?analysis",
    re.I,
)


def review_hint(title, abstract):
    """Return True if title/abstract text suggests a review or meta-analysis manuscript.

    Heuristic only — used to suggest a checkbox default in the UI, never fed to the model.
    """
    text = f"{title or ''} {abstract or ''}"
    return bool(REVIEW_HINT_RX.search(text))


@lru_cache(maxsize=1)
def _models():
    return (
        joblib.load(os.path.join(MODELS, "model_B_final.pkl")),
        joblib.load(os.path.join(MODELS, "model_A_final.pkl")),
        joblib.load(os.path.join(MODELS, "reg_c3_final.pkl")),
        joblib.load(os.path.join(MODELS, "text_artifacts_final.pkl")),
    )


@lru_cache(maxsize=1)
def _encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def base_feats(title, abstract, n_references, is_review):
    title = title or ""
    abstract = abstract or ""
    return {
        "n_authors": C["n_authors"], "n_institutions": C["n_institutions"],
        "n_countries": 1, "is_international": 0, "has_korea": 0, "has_usa": 0,
        "title_n_words": len(title.split()), "title_n_chars": len(title),
        "title_has_colon": int(":" in title), "title_has_question": int("?" in title),
        "abstract_n_words": len(abstract.split()), "has_abstract": int(bool(abstract)),
        "n_references": n_references, "is_oa": 1, "is_review": int(is_review),
        "pub_month": 6, "journal": "Asian Spine Journal", "topic_field": "Medicine",
        "topic_subfield": "Orthopedics and Sports Medicine", "oa_status": "gold",
    }


def features(title, abstract, n_references, is_review):
    _, _, _, art = _models()
    title = title or ""
    abstract = abstract or ""
    text = (title + " " + abstract).strip()
    f = base_feats(title, abstract, n_references, is_review)
    f.update(C["au"])
    for k, r in RX.items():
        f[k] = int(bool(re.search(r, text, re.I)))
    try:
        import textstat
        f["flesch"] = round(textstat.flesch_reading_ease(abstract), 1) if abstract else C["flesch_med"]
    except Exception:
        f["flesch"] = C["flesch_med"]
    f["n_topics"] = C["n_topics_med"]
    f["subfield_size"] = C["subfield_size_med"]
    f["subfield_year_count"] = C["subfield_year_count_med"]
    f["subfield_growth"] = C["subfield_growth_med"]
    e = _encoder().encode([title + ". " + abstract], normalize_embeddings=True)
    red = art["pca"].transform(e)[0]
    for j in range(50):
        f[f"t_{j}"] = float(red[j])
    f["nov_c"] = float(1 - (normalize(e) @ art["centroid"].T).ravel()[0])
    return f


def _proba(pack, f):
    clf, num, cat = pack["model"], pack["num"], pack["cat"]
    X = pd.DataFrame([{c: f.get(c, 0) for c in num + cat}])
    for c in cat:
        X[c] = X[c].astype("category")
    return float(clf.predict_proba(X)[0, 1])


def band(p):
    """Return (label, color) for a within-journal top-25% probability (baseline 25%)."""
    if p >= 0.40:
        return "상위권 가능성 높음", "#1a7f37"
    if p >= 0.30:
        return "평균 이상", "#1f6feb"
    if p >= 0.18:
        return "평균 수준", "#9a6700"
    return "평균 이하", "#cf222e"


def explain(pack, f, pbase):
    """Occlusion: replace each interpretable factor with a corpus baseline → probability change."""
    clf, num, cat = pack["model"], pack["num"], pack["cat"]
    base = {
        "n_references": C["n_references_med"], "abstract_n_words": C["abstract_n_words_med"],
        "is_review": 0, "is_sr_meta": 0, "is_rct": 0, "nov_c": C["nov_c_med"],
    }
    labels = {
        "n_references": "참고문헌 수", "abstract_n_words": "초록 길이", "is_review": "리뷰 논문 여부",
        "is_sr_meta": "메타분석/SR 여부", "is_rct": "RCT 여부", "nov_c": "주제 신규성",
    }
    out = []
    for k, bv in base.items():
        if k not in num:
            continue
        g = dict(f)
        g[k] = bv
        X = pd.DataFrame([{c: g.get(c, 0) for c in num + cat}])
        for c in cat:
            X[c] = X[c].astype("category")
        out.append((labels[k], pbase - float(clf.predict_proba(X)[0, 1])))
    out.sort(key=lambda x: -abs(x[1]))
    return out


def score(title, abstract, n_references, is_review=False):
    """Return a dict of interpretable results for one manuscript."""
    clfB, clfA, reg, _ = _models()
    f = features(title, abstract, n_references, is_review)
    probB = _proba(clfB, f)
    probA = _proba(clfA, f)
    rg, rn, rc = reg["model"], reg["num"], reg["cat"]
    Xr = pd.DataFrame([{c: f.get(c, 0) for c in rn + rc}])
    for c in rc:
        Xr[c] = Xr[c].astype("category")
    c3 = float(np.expm1(rg.predict(Xr)[0]))
    label, color = band(probB)
    return {
        "prob_asj_top25": probB,
        "prob_field_top10": probA,
        "cite3y": max(0.0, c3),
        "cite3y_lo": max(0, round(c3 - 4)),
        "cite3y_hi": round(c3 + 4),
        "asj_c3_med": C["asj_c3_med"],
        "all_c3_med": C["all_c3_med"],
        "band": label,
        "band_color": color,
        "drivers": explain(clfB, f, probB),
    }
