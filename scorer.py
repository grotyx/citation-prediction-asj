# -*- coding: utf-8 -*-
"""Citation-propensity scorer for spine articles (REV1 locked reduced predictor set).

Uses the same locked models reported as primary in the revised manuscript: Model B
(within-journal top quartile) and Model A (field top decile), trained on 2018-2021 and
applied unchanged. The reduced predictor set deliberately excludes variables that OpenAlex
records only after publication (topic and subfield annotations and their derivatives,
open-access status, current last-author h-index) and temporally unsafe author metrics.

The score estimates citability, not scientific merit or clinical value.
"""
import json
import os
import re
from functools import lru_cache

import joblib
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
C = json.load(open(os.path.join(MODELS, "constants_reduced.json"), encoding="utf-8"))
MED = C["medians"]

# text-derived study-design flags — the keyword rules used in training
RX = {
    "is_rct": r"\brandomi[sz]ed|\bRCT\b",
    "is_sr_meta": r"meta-?analysis|systematic review",
    "is_cohort": r"\bcohort\b|prospective|retrospective",
    "is_case_report": r"\bcase report|case series",
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
        joblib.load(os.path.join(MODELS, "model_B_reduced.pkl")),
        joblib.load(os.path.join(MODELS, "model_A_reduced.pkl")),
    )


@lru_cache(maxsize=1)
def _encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(C["embedding_model"], revision=C["embedding_revision"], device="cpu")


def features(title, abstract, n_references, is_review, n_authors=None, n_institutions=None,
             n_countries=None):
    """Build the 18 reduced numeric features. Unknown author history falls back to corpus
    medians; abstract-derived features stay missing when no abstract is supplied, which is
    how the model was trained (native missing-value handling, never encoded as zero)."""
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    has_abstract = int(bool(abstract))
    text = (title + " " + abstract).strip()

    if abstract:
        try:
            import textstat

            flesch = round(textstat.flesch_reading_ease(abstract), 1)
        except Exception:
            flesch = MED["flesch"]
        abstract_n_words = len(abstract.split())
    else:
        flesch = np.nan
        abstract_n_words = np.nan

    n_countries = MED["n_countries"] if n_countries is None else float(n_countries)
    f = {
        "n_authors": MED["n_authors"] if n_authors is None else float(n_authors),
        "n_institutions": MED["n_institutions"] if n_institutions is None else float(n_institutions),
        "n_countries": n_countries,
        "is_international": float(n_countries > 1),
        "is_review": float(int(is_review)),
        "title_n_words": float(len(title.split())),
        "abstract_n_words": abstract_n_words,
        "has_abstract": float(has_abstract),
        "flesch": flesch,
        "n_references": float(n_references),
        "fa_prior_works_log1p": float(np.log1p(MED["fa_prior_works"])),
        "fa_prior_cites_log1p": float(np.log1p(MED["fa_prior_cites"])),
        "la_prior_works_log1p": float(np.log1p(MED["la_prior_works"])),
        "la_prior_cites_log1p": float(np.log1p(MED["la_prior_cites"])),
    }
    for k, r in RX.items():
        f[k] = float(bool(re.search(r, text, re.I)))
    return f, text


def _design(pack, f, embedding, journal="Asian Spine Journal"):
    """Reproduce LockedModel.transform: numeric block, then PCA block, then categoricals."""
    parts = [np.array([[f[c] for c in pack["numeric"]]], dtype=float)]
    parts.append(pack["pca"].transform(embedding))
    for column in pack["categorical"]:
        mapping = pack["category_maps"][column]
        value = journal if column == "journal" else "__MISSING__"
        parts.append(np.array([[float(mapping.get(str(value), -1))]]))
    return np.concatenate(parts, axis=1)


def _proba(pack, f, embedding, journal):
    return float(pack["classifier"].predict_proba(_design(pack, f, embedding, journal))[0, 1])


def band(p):
    """Label for a within-journal top-quartile probability (baseline ~25%)."""
    if p >= 0.40:
        return "상위권 가능성 높음", "#1a7f37"
    if p >= 0.30:
        return "평균 이상", "#1f6feb"
    if p >= 0.18:
        return "평균 수준", "#9a6700"
    return "평균 이하", "#cf222e"


def explain(pack, f, embedding, journal, pbase):
    """Occlusion sensitivity: replace one input with a corpus baseline and record the change.

    These are model sensitivities, not causal effects, and not advice to authors."""
    baseline = {
        "n_references": MED["n_references"],
        "abstract_n_words": MED["abstract_n_words"],
        "is_review": 0.0,
        "is_sr_meta": 0.0,
        "is_rct": 0.0,
        "title_n_words": MED["title_n_words"],
    }
    labels = {
        "n_references": "참고문헌 수",
        "abstract_n_words": "초록 길이",
        "is_review": "리뷰 논문 여부",
        "is_sr_meta": "메타분석/SR 여부",
        "is_rct": "RCT 여부",
        "title_n_words": "제목 길이",
    }
    out = []
    for k, bv in baseline.items():
        if k not in pack["numeric"]:
            continue
        g = dict(f)
        g[k] = bv
        out.append((labels[k], pbase - _proba(pack, g, embedding, journal)))
    out.sort(key=lambda x: -abs(x[1]))
    return out


def score(title, abstract, n_references, is_review=False, journal="Asian Spine Journal",
          n_authors=None, n_institutions=None, n_countries=None):
    """Return interpretable results for one article."""
    packB, packA = _models()
    f, text = features(title, abstract, n_references, is_review, n_authors, n_institutions,
                       n_countries)
    embedding = np.asarray(
        _encoder().encode([text], normalize_embeddings=True), dtype=np.float32
    )
    probB = _proba(packB, f, embedding, journal)
    probA = _proba(packA, f, embedding, journal)
    label, color = band(probB)
    return {
        "prob_injournal_top25": probB,
        "prob_field_top10": probA,
        "base_rate_top25": C["event_rate"]["model_B_within_journal_top25"],
        "base_rate_top10": C["event_rate"]["model_A_field_top10"],
        "band": label,
        "band_color": color,
        "drivers": explain(packB, f, embedding, journal, probB),
        "predictor_set": C["predictor_set"],
    }


def _demo():
    """Smallest runnable check: the design matrix must match what the model was fitted on,
    probabilities must be valid, and a missing abstract must stay missing (not zero)."""
    packB, packA = _models()
    f, _ = features("A randomized controlled trial of lumbar fusion", "word " * 250, 40, False)
    assert set(packB["numeric"]) <= set(f), "reduced feature set mismatch"
    assert f["is_rct"] == 1.0 and f["is_review"] == 0.0
    nf, _ = features("Title only", "", 30, False)
    assert np.isnan(nf["abstract_n_words"]) and np.isnan(nf["flesch"]), "missing abstract encoded as zero"
    assert nf["has_abstract"] == 0.0
    assert review_hint("A systematic review of fusion", "") and not review_hint("A cohort study", "")
    n_expected = len(packB["numeric"]) + 50 + len(packB["categorical"])
    emb = np.zeros((1, 384), dtype=np.float32)
    assert _design(packB, f, emb).shape == (1, n_expected), "design matrix width mismatch"
    assert _design(packA, f, emb).shape == (1, len(packA["numeric"]) + 50 + len(packA["categorical"]))
    r = score("A randomized controlled trial of lumbar fusion", "word " * 250, 40, False)
    assert 0.0 <= r["prob_injournal_top25"] <= 1.0 and 0.0 <= r["prob_field_top10"] <= 1.0
    print("scorer self-check OK:", {k: round(v, 3) for k, v in r.items() if isinstance(v, float)})


if __name__ == "__main__":
    _demo()
