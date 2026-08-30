from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy.special import expit
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PLAN = DATA / "analysis_plan.md"
MANIFEST = DATA / "rev1_reproducibility_manifest.json"
SCRIPT = Path(__file__).resolve()
PLAN_SHA256 = "e17faa968e38c7fc622349840d4b5f9f8cdbaafdefb4ec8af883a59d9b245b4b"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SEED = 42
BOOTSTRAPS = 1000
TRAIN_YEARS = (2018, 2019, 2020, 2021)
VALID_YEARS = (2022, 2023)

PROV = ["freeze_id", "plan_sha256", "source_cohort_sha256", "analysis_script_sha256", "generated_at"]
OUTPUT_COLUMNS = {
    "rev1_predictor_availability.csv": PROV + ["predictor", "feature_group", "source_file", "source_field", "actual_measurement_time", "conceptual_preacceptance_availability", "leakage_risk", "reduced_model_inclusion", "expanded_model_inclusion", "model_scope", "transformation", "missing_data_rule", "exclusion_reason", "notes"],
    "rev1_model_performance.csv": PROV + ["model_id", "model_role", "predictor_set", "sensitivity", "algorithm", "calibration_method", "training_years", "validation_year", "population", "n", "events", "non_events", "event_rate", "metric", "estimate", "ci_lower", "ci_upper", "bootstrap_requested", "bootstrap_valid", "random_seed", "comparator", "difference_direction", "notes"],
    "rev1_abstract_checks.csv": PROV + ["analysis_type", "model_id", "predictor_set", "training_population", "validation_year", "document_type", "abstract_status", "n", "denominator_n", "events", "non_events", "event_rate", "metric", "estimate", "ci_lower", "ci_upper", "bootstrap_requested", "bootstrap_valid", "test_name", "test_statistic", "p_value", "notes"],
    "rev1_table2_models.csv": PROV + ["analysis_id", "row_type", "population", "model_variant", "outcome", "term", "term_role", "raw_scale", "transformation", "standardization_mean", "standardization_sd", "n", "events", "non_events", "top10_group_n", "other_group_n", "missing_excluded_n", "estimate_type", "estimate", "ci_lower", "ci_upper", "p_value", "converged", "separation_or_rank_issue", "notes"],
    "rev1_threshold_performance.csv": PROV + ["model_id", "predictor_set", "cv_scheme", "operating_point", "target_flagged_per_100", "training_probability_cutoff", "training_actual_flagged_per_100", "validation_year", "n", "events", "non_events", "metric", "numerator", "denominator", "estimate", "ci_lower", "ci_upper", "bootstrap_requested", "bootstrap_valid", "random_seed", "notes"],
    "rev1_ablation.csv": PROV + ["model_id", "predictor_set", "algorithm", "calibration_method", "training_years", "validation_year", "feature_step", "feature_groups_included", "n", "events", "non_events", "roc_auc", "ci_lower", "ci_upper", "delta_from_prior_step", "table3_comparator_key", "table3_roc_auc", "consistency_status", "notes"],
    "rev1_descriptive_checks.csv": PROV + ["analysis_type", "grouping_variable", "group_level", "document_type", "publication_year", "journal", "model_id", "variable", "statistic", "n", "denominator_n", "events", "non_events", "estimate", "q1", "q3", "rate", "test_name", "test_statistic", "p_value", "effect_term", "effect_estimate", "ci_lower", "ci_upper", "tie_method", "estimable", "notes"],
}

REDUCED_NUMERIC = [
    "n_authors", "n_institutions", "n_countries", "is_international", "is_review",
    "title_n_words", "abstract_n_words", "has_abstract", "flesch", "is_rct",
    "is_sr_meta", "is_cohort", "is_case_report", "fa_prior_works_log1p",
    "fa_prior_cites_log1p", "la_prior_works_log1p", "la_prior_cites_log1p",
    "n_references",
]
EXPANDED_NUMERIC = [
    "n_authors", "n_institutions", "n_countries", "is_international", "has_korea",
    "has_usa", "title_n_words", "title_n_chars", "title_has_colon",
    "title_has_question", "abstract_n_words", "has_abstract", "n_references", "is_oa",
    "is_review", "fa_prior_works_log1p", "fa_prior_cites_log1p",
    "la_prior_works_log1p", "la_prior_cites_log1p", "team_prior_works_max",
    "team_prior_cites_max", "la_h_index_now", "is_rct", "is_sr_meta", "is_cohort",
    "is_case_report", "sw_novel", "sw_ai", "sw_guideline", "n_topics", "flesch",
    "subfield_size", "subfield_year_count", "subfield_growth",
]
AUTHOR_SAFE = ["fa_prior_works", "fa_prior_cites", "la_prior_works", "la_prior_cites"]
AUTHOR_LOG = [f"{c}_log1p" for c in AUTHOR_SAFE]
TABLE2_CONTINUOUS = ["n_references", "n_authors", "n_institutions", "abstract_n_words", *AUTHOR_SAFE]
TABLE2_BINARY = ["is_international", "is_oa", "is_review", "is_rct", "is_sr_meta", "has_usa"]
TIE_METHOD = "pandas rank(method='average', pct=True); positive when percentile >= cutoff"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_plan_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    canonical, count = re.subn(
        r"<!-- APPROVAL-METADATA-BEGIN -->.*?<!-- APPROVAL-METADATA-END -->",
        "<!-- APPROVAL-METADATA-EXCLUDED -->",
        text,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise RuntimeError("approval metadata block is missing or ambiguous")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def provenance(manifest: dict, generated_at: str) -> dict:
    return {
        "freeze_id": manifest["freeze_id"],
        "plan_sha256": PLAN_SHA256,
        "source_cohort_sha256": manifest["source_cohort"]["sha256"],
        "analysis_script_sha256": sha256(SCRIPT),
        "generated_at": generated_at,
    }


def frame(rows: list[dict], name: str, prov: dict) -> pd.DataFrame:
    columns = OUTPUT_COLUMNS[name]
    out = pd.DataFrame([{**prov, **row} for row in rows])
    for column in columns:
        if column not in out:
            out[column] = np.nan
    return out[columns]


def reconstruct_author_availability(cohort_ids: set[str]) -> pd.DataFrame:
    rows: list[dict] = []
    seen: set[str] = set()
    with (DATA / "raw_works.jsonl").open(encoding="utf-8") as f:
        for line in f:
            work = json.loads(line)
            work_id = (work.get("id") or "").split("/")[-1]
            if work_id not in cohort_ids or work_id in seen:
                continue
            seen.add(work_id)
            authorships = work.get("authorships") or []
            first = last = None
            for authorship in authorships:
                author_id = (((authorship.get("author") or {}).get("id") or "").split("/")[-1] or None)
                if authorship.get("author_position") == "first" and first is None:
                    first = author_id
                if authorship.get("author_position") == "last":
                    last = author_id
            if first is None and authorships:
                first = ((((authorships[0].get("author") or {}).get("id") or "").split("/")[-1]) or None)
            if last is None and authorships:
                last = ((((authorships[-1].get("author") or {}).get("id") or "").split("/")[-1]) or None)
            rows.append({"id": work_id, "first_author_id_available": first is not None, "last_author_id_available": last is not None})
    if len(rows) != len(cohort_ids):
        raise RuntimeError("raw source cannot reproduce author-ID availability for every cohort work")
    return pd.DataFrame(rows)


def load_data(manifest: dict) -> pd.DataFrame:
    source = DATA / "features.parquet"
    if sha256(source) != manifest["source_cohort"]["sha256"]:
        raise RuntimeError("source cohort hash changed")
    df = pd.read_parquet(source)
    if len(df) != 13299 or df["id"].nunique() != 13299 or df["id"].duplicated().any():
        raise RuntimeError("frozen cohort identity does not match the approved population")
    a = pd.read_parquet(DATA / "author_features.parquet")
    v3 = pd.read_parquet(DATA / "features_v3extra.parquet")
    df = df.merge(a, on="id", how="left", validate="one_to_one").merge(v3, on="id", how="left", validate="one_to_one")
    availability = reconstruct_author_availability(set(df["id"]))
    df = df.merge(availability, on="id", how="left", validate="one_to_one")

    # Restore source unavailability to missing without altering the frozen source files.
    df.loc[~df["first_author_id_available"], ["fa_prior_works", "fa_prior_cites"]] = np.nan
    df.loc[~df["last_author_id_available"], ["la_prior_works", "la_prior_cites", "la_h_index_now"]] = np.nan
    both_missing = ~df["first_author_id_available"] & ~df["last_author_id_available"]
    df.loc[both_missing, ["team_prior_works_max", "team_prior_cites_max"]] = np.nan
    for column in AUTHOR_SAFE:
        if (df[column].dropna() < 0).any():
            raise RuntimeError(f"negative prior-author count: {column}")
        df[f"{column}_log1p"] = np.log1p(df[column])

    no_abstract = df["has_abstract"].ne(1)
    df.loc[no_abstract, ["abstract_n_words", "flesch"]] = np.nan

    live_a = df.groupby("pub_year")["cited_by_count"].rank(method="average", pct=True).ge(0.90).astype(int)
    live_b = df.groupby(["journal", "pub_year"])["cited_by_count"].rank(method="average", pct=True).ge(0.75).astype(int)
    if not live_a.equals(df["is_top10"].astype(int)) or not live_b.equals(df["is_top25_jrnl"].astype(int)):
        raise RuntimeError("live label reproduction disagrees with the frozen labels")
    if set(df["pub_year"].unique()) != set((*TRAIN_YEARS, *VALID_YEARS)):
        raise RuntimeError("frozen years disagree with the approved plan")
    return df


@dataclass
class LockedModel:
    numeric: list[str]
    categorical: list[str]
    use_embedding: bool
    use_novelty: bool = False
    pca: PCA | None = None
    centroid: np.ndarray | None = None
    category_maps: dict[str, dict[str, int]] | None = None
    classifier: CalibratedClassifierCV | None = None

    def fit(self, df: pd.DataFrame, embeddings: np.ndarray, y: np.ndarray) -> "LockedModel":
        self.pca = PCA(n_components=50, random_state=SEED).fit(embeddings) if self.use_embedding else None
        self.centroid = normalize(embeddings.mean(axis=0, keepdims=True)) if self.use_novelty else None
        self.category_maps = {}
        for column in self.categorical:
            values = df[column].fillna("__MISSING__").astype(str)
            categories = sorted(values.unique().tolist())
            if len(categories) > 255:
                raise RuntimeError(f"native categorical cardinality exceeds 255: {column}")
            self.category_maps[column] = {value: i for i, value in enumerate(categories)}
        x = self.transform(df, embeddings)
        n_categorical = len(self.categorical)
        categorical_mask = [False] * (x.shape[1] - n_categorical) + [True] * n_categorical
        base = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            categorical_features=categorical_mask,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=SEED,
        )
        self.classifier = CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)
        self.classifier.fit(x, y)
        return self

    def transform(self, df: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
        parts = [df[self.numeric].to_numpy(dtype=float)] if self.numeric else []
        if self.use_embedding:
            if self.pca is None:
                raise RuntimeError("PCA is not fitted")
            parts.append(self.pca.transform(embeddings))
        if self.use_novelty:
            if self.centroid is None:
                raise RuntimeError("novelty centroid is not fitted")
            parts.append(1 - normalize(embeddings) @ self.centroid.T)
        for column in self.categorical:
            mapping = (self.category_maps or {})[column]
            values = df[column].fillna("__MISSING__").astype(str).map(mapping).fillna(-1).to_numpy(dtype=float)
            parts.append(values[:, None])
        if not parts:
            raise RuntimeError("empty design matrix")
        return np.concatenate(parts, axis=1)

    def predict(self, df: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("classifier is not fitted")
        return self.classifier.predict_proba(self.transform(df, embeddings))[:, 1]


def fit_locked(df: pd.DataFrame, embeddings: np.ndarray, y: np.ndarray, numeric: list[str], categorical: list[str], use_embedding: bool, use_novelty: bool = False) -> LockedModel:
    return LockedModel(numeric, categorical, use_embedding, use_novelty).fit(df, embeddings, y)


def calibration_coefficients(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    if np.unique(y).size < 2:
        return np.nan, np.nan
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    x = np.column_stack([np.ones(len(y)), z])
    beta = np.array([0.0, 1.0])
    for _ in range(50):
        mu = expit(x @ beta)
        w = np.clip(mu * (1 - mu), 1e-9, None)
        hessian = x.T @ (w[:, None] * x)
        step = np.linalg.solve(hessian, x.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    intercept = 0.0
    for _ in range(50):
        mu = expit(intercept + z)
        step = (y - mu).sum() / np.clip((mu * (1 - mu)).sum(), 1e-9, None)
        intercept += step
        if abs(step) < 1e-9:
            break
    return float(beta[1]), float(intercept)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    slope, citl = calibration_coefficients(y, p)
    return {
        "roc_auc": roc_auc_score(y, p) if np.unique(y).size == 2 else np.nan,
        "pr_auc": average_precision_score(y, p) if np.unique(y).size == 2 else np.nan,
        "brier_score": brier_score_loss(y, p),
        "calibration_slope": slope,
        "calibration_in_the_large": citl,
    }


def bootstrap_metrics(y: np.ndarray, p: np.ndarray, requested: int = BOOTSTRAPS) -> dict[str, tuple[float, float, int]]:
    rng = np.random.default_rng(SEED)
    values = {name: [] for name in metrics(y, p)}
    for _ in range(requested):
        idx = rng.integers(0, len(y), len(y))
        try:
            sample = metrics(y[idx], p[idx])
        except (ValueError, np.linalg.LinAlgError, OverflowError):
            continue
        for name, value in sample.items():
            if np.isfinite(value):
                values[name].append(value)
    return {
        name: (*np.percentile(items, [2.5, 97.5]), len(items)) if items else (np.nan, np.nan, 0)
        for name, items in values.items()
    }


def bootstrap_differences(y: np.ndarray, p_left: np.ndarray, p_right: np.ndarray) -> dict[str, tuple[float, float, int]]:
    rng = np.random.default_rng(SEED)
    values = {name: [] for name in metrics(y, p_left)}
    for _ in range(BOOTSTRAPS):
        idx = rng.integers(0, len(y), len(y))
        try:
            left, right = metrics(y[idx], p_left[idx]), metrics(y[idx], p_right[idx])
        except (ValueError, np.linalg.LinAlgError, OverflowError):
            continue
        for name in values:
            difference = left[name] - right[name]
            if np.isfinite(difference):
                values[name].append(difference)
    return {
        name: (*np.percentile(items, [2.5, 97.5]), len(items)) if items else (np.nan, np.nan, 0)
        for name, items in values.items()
    }


def predictor_rows() -> list[dict]:
    rows: list[dict] = []

    def add(predictor: str, group: str, source_file: str, source_field: str, timing: str, availability: str, risk: str, reduced: str, expanded: str, scope: str = "Model A and Model B", transformation: str = "none", missing: str = "native numeric missing handling", exclusion: str = "", notes: str = "") -> None:
        rows.append({"predictor": predictor, "feature_group": group, "source_file": source_file, "source_field": source_field, "actual_measurement_time": timing, "conceptual_preacceptance_availability": availability, "leakage_risk": risk, "reduced_model_inclusion": reduced, "expanded_model_inclusion": expanded, "model_scope": scope, "transformation": transformation, "missing_data_rule": missing, "exclusion_reason": exclusion, "notes": notes})

    reduced = set(REDUCED_NUMERIC)
    expanded = set(EXPANDED_NUMERIC)
    groups = {
        "n_authors": "team/collaboration", "n_institutions": "team/collaboration", "n_countries": "team/collaboration", "is_international": "team/collaboration", "has_korea": "country indicator", "has_usa": "country indicator",
        "title_n_words": "document/text structure", "title_n_chars": "document/text structure", "title_has_colon": "document/text structure", "title_has_question": "document/text structure", "abstract_n_words": "document/text structure", "has_abstract": "document/text structure", "n_references": "reference-list proxy", "is_oa": "open access", "is_review": "document/text structure",
        "fa_prior_works_log1p": "author prominence", "fa_prior_cites_log1p": "author prominence", "la_prior_works_log1p": "author prominence", "la_prior_cites_log1p": "author prominence", "team_prior_works_max": "author prominence", "team_prior_cites_max": "author prominence", "la_h_index_now": "author prominence",
        "is_rct": "text-derived study design", "is_sr_meta": "text-derived study design", "is_cohort": "text-derived study design", "is_case_report": "text-derived study design", "sw_novel": "study/topic annotation", "sw_ai": "study/topic annotation", "sw_guideline": "study/topic annotation", "n_topics": "OpenAlex topic annotation", "flesch": "document/text structure", "subfield_size": "OpenAlex topic derivative", "subfield_year_count": "OpenAlex topic derivative", "subfield_growth": "OpenAlex topic derivative",
    }
    source_fields = {name: name.replace("_log1p", "") for name in groups}
    author_fields = {f"{c}_log1p" for c in AUTHOR_SAFE} | {"team_prior_works_max", "team_prior_cites_max", "la_h_index_now"}
    v3_fields = {"is_rct", "is_sr_meta", "is_cohort", "is_case_report", "sw_novel", "sw_ai", "sw_guideline", "n_topics", "flesch", "subfield_size", "subfield_year_count", "subfield_growth"}
    for name, group in groups.items():
        source_file = "data/author_features.parquet" if name in author_fields else "data/features_v3extra.parquet" if name in v3_fields else "data/features.parquet"
        timing = "OpenAlex author counts through calendar year preceding index publication" if name in author_fields - {"la_h_index_now"} else "current OpenAlex author summary at source retrieval" if name == "la_h_index_now" else "publication-record metadata at source retrieval"
        availability = "yes, as publication-record proxy" if name in reduced else "no or not retained in reduced model"
        risk = "post-index accumulation" if name == "la_h_index_now" else "post-publication annotation" if group.startswith("OpenAlex") or name == "is_oa" else "proxy/change-during-review" if name == "n_references" else "low within frozen source definition"
        transform = "log1p" if name.endswith("_log1p") else "none"
        exclusion = "" if name in reduced else "not in locked reduced set"
        add(name.replace("_log1p", ""), group, source_file, source_fields[name], timing, availability, risk, str(name in reduced).lower(), str(name in expanded).lower(), transformation=transform, exclusion=exclusion)
    for i in range(1, 51):
        add(f"embedding_pc_{i:02d}", "text content", "data/features.parquet", "title + abstract", "publication-record text at source retrieval", "yes, as publication-record proxy", "proxy/change-during-review", "true", "true", transformation="normalized all-MiniLM-L6-v2 embedding; training-only 50-component PCA", missing="title-only text when abstract absent")
    add("training_centroid_text_novelty", "text content", "data/features.parquet", "title + abstract", "derived at model fitting", "no", "expanded publication-metadata comparator only", "false", "true", transformation="1 - cosine similarity to training centroid", exclusion="not in locked reduced set")
    for name in ["journal", "topic_field", "topic_subfield", "oa_status"]:
        add(name, "categorical metadata", "data/features.parquet", name, "publication-record/OpenAlex annotation at source retrieval", "journal only for Model A" if name == "journal" else "no", "post-publication annotation" if name != "journal" else "journal identity", "Model A only" if name == "journal" else "false", "Model A only" if name == "journal" else "true", scope="Model A only" if name == "journal" else "Model A and Model B", missing="explicit missing category learned from training", exclusion="not in reduced set" if name != "journal" else "excluded from Model B by endpoint design")
    excluded = [
        ("pub_month", "publication timing", "data/features.parquet", "publication month unavailable before acceptance"),
        ("ref_mean_age", "reference-network", "data/features_v3refs.parquet", "reference-derived feature excluded"),
        ("ref_pct_recent5", "reference-network", "data/features_v3refs.parquet", "reference-derived feature excluded"),
        ("ref_mean_logcit", "reference citation impact", "data/features_v3refs.parquet", "post-publication citation leakage"),
        ("ref_max_cit", "reference citation impact", "data/features_v3refs.parquet", "post-publication citation leakage"),
        ("ref_n_resolved", "reference-network", "data/features_v3refs.parquet", "resolution-count variable excluded"),
        ("cited_by_count", "outcome proxy", "data/features.parquet", "index-article post-publication outcome"),
        ("c2_early", "outcome proxy", "data/features.parquet", "index-article early citation outcome"),
        ("oa_norm_percentile", "outcome proxy", "data/features.parquet", "post-publication normalized citation percentile"),
        ("fwci", "outcome proxy", "data/features.parquet", "post-publication citation impact"),
    ]
    for name, group, source_file, reason in excluded:
        add(name, group, source_file, name, "source retrieval or post-publication", "no", "high or unapproved", "false", "false", exclusion=reason)
    return rows


def model_categories(model_id: str, expanded: bool = False) -> list[str]:
    categories = ["topic_field", "topic_subfield", "oa_status"] if expanded else []
    if model_id == "A":
        categories = ["journal", *categories]
    return categories


def performance_rows(df: pd.DataFrame, embeddings: np.ndarray) -> tuple[list[dict], dict, dict]:
    rows: list[dict] = []
    fitted: dict = {}
    predictions: dict = {}
    train_mask = df["pub_year"].isin(TRAIN_YEARS).to_numpy()
    train = df.loc[train_mask]
    emb_train = embeddings[train_mask]
    for model_id, label, role in [("B", "is_top25_jrnl", "primary"), ("A", "is_top10", "secondary")]:
        specs = [
            ("reduced", "primary", REDUCED_NUMERIC, model_categories(model_id), True, False),
            ("expanded", "comparator", EXPANDED_NUMERIC, model_categories(model_id, expanded=True), True, True),
            ("reduced_without_reference", "without_reference", [c for c in REDUCED_NUMERIC if c != "n_references"], model_categories(model_id), True, False),
        ]
        for predictor_set, sensitivity, numeric, categorical, use_embedding, use_novelty in specs:
            fitted[(model_id, predictor_set)] = fit_locked(train, emb_train, train[label].to_numpy(int), numeric, categorical, use_embedding, use_novelty)
            for year in VALID_YEARS:
                mask = df["pub_year"].eq(year).to_numpy()
                valid = df.loc[mask]
                p = fitted[(model_id, predictor_set)].predict(valid, embeddings[mask])
                predictions[(model_id, predictor_set, year)] = p
                y = valid[label].to_numpy(int)
                point, cis = metrics(y, p), bootstrap_metrics(y, p)
                for metric, estimate in point.items():
                    lo, hi, valid_boot = cis[metric]
                    rows.append({"model_id": model_id, "model_role": role, "predictor_set": predictor_set, "sensitivity": sensitivity, "algorithm": "histogram gradient boosting", "calibration_method": "isotonic, 3-fold training CV", "training_years": "2018-2021", "validation_year": year, "population": f"P_valid_{year}", "n": len(valid), "events": int(y.sum()), "non_events": int(len(y) - y.sum()), "event_rate": y.mean(), "metric": metric, "estimate": estimate, "ci_lower": lo, "ci_upper": hi, "bootstrap_requested": BOOTSTRAPS, "bootstrap_valid": valid_boot, "random_seed": SEED, "comparator": "", "difference_direction": "", "notes": "separate held-out-year evaluation"})
        for year in VALID_YEARS:
            mask = df["pub_year"].eq(year).to_numpy()
            y = df.loc[mask, label].to_numpy(int)
            contrasts = [
                ("reduced_minus_expanded", "reduced", "expanded"),
                ("reduced_without_reference_minus_reduced", "reduced_without_reference", "reduced"),
            ]
            for contrast, left, right in contrasts:
                p_left, p_right = predictions[(model_id, left, year)], predictions[(model_id, right, year)]
                point_left, point_right = metrics(y, p_left), metrics(y, p_right)
                cis = bootstrap_differences(y, p_left, p_right)
                for metric in point_left:
                    lo, hi, valid_boot = cis[metric]
                    rows.append({"model_id": model_id, "model_role": role, "predictor_set": contrast, "sensitivity": "paired_difference", "algorithm": "histogram gradient boosting", "calibration_method": "isotonic, 3-fold training CV", "training_years": "2018-2021", "validation_year": year, "population": f"P_valid_{year}", "n": len(y), "events": int(y.sum()), "non_events": int(len(y) - y.sum()), "event_rate": y.mean(), "metric": metric, "estimate": point_left[metric] - point_right[metric], "ci_lower": lo, "ci_upper": hi, "bootstrap_requested": BOOTSTRAPS, "bootstrap_valid": valid_boot, "random_seed": SEED, "comparator": right, "difference_direction": f"{left} minus {right}", "notes": "paired article-level bootstrap"})
    return rows, fitted, predictions


def abstract_rows(df: pd.DataFrame, embeddings: np.ndarray, fitted: dict) -> list[dict]:
    rows: list[dict] = []
    for year in VALID_YEARS:
        year_df = df[df["pub_year"].eq(year)]
        for document_type, type_mask in [("Original article", year_df["is_review"].eq(0)), ("Review", year_df["is_review"].eq(1))]:
            denominator = int(type_mask.sum())
            for status, value in [("present", 1), ("absent", 0)]:
                n = int((type_mask & year_df["has_abstract"].eq(value)).sum())
                rows.append({"analysis_type": "availability", "model_id": "", "predictor_set": "", "training_population": "", "validation_year": year, "document_type": document_type, "abstract_status": status, "n": n, "denominator_n": denominator, "metric": "proportion", "estimate": n / denominator if denominator else np.nan, "bootstrap_requested": 0, "bootstrap_valid": 0, "notes": "machine-readable OpenAlex abstract"})
    for model_id, label in [("B", "is_top25_jrnl"), ("A", "is_top10")]:
        for year in VALID_YEARS:
            for status, value in [("present", 1), ("absent", 0)]:
                mask = df["pub_year"].eq(year) & df["has_abstract"].eq(value)
                part = df.loc[mask]
                y = part[label].to_numpy(int)
                p = fitted[(model_id, "reduced")].predict(part, embeddings[mask.to_numpy()])
                estimate = roc_auc_score(y, p) if np.unique(y).size == 2 else np.nan
                if np.isfinite(estimate):
                    ci = bootstrap_metrics(y, p)["roc_auc"]
                else:
                    ci = (np.nan, np.nan, 0)
                rows.append({"analysis_type": "locked_all_record_subgroup", "model_id": model_id, "predictor_set": "reduced", "training_population": "P_train", "validation_year": year, "document_type": "All", "abstract_status": status, "n": len(part), "denominator_n": len(part), "events": int(y.sum()), "non_events": int(len(y) - y.sum()), "event_rate": y.mean() if len(y) else np.nan, "metric": "roc_auc", "estimate": estimate, "ci_lower": ci[0], "ci_upper": ci[1], "bootstrap_requested": BOOTSTRAPS, "bootstrap_valid": ci[2], "notes": "NA when subgroup lacks both outcome classes"})
        train_mask = df["pub_year"].isin(TRAIN_YEARS) & df["has_abstract"].eq(1)
        train = df.loc[train_mask]
        model = fit_locked(train, embeddings[train_mask.to_numpy()], train[label].to_numpy(int), REDUCED_NUMERIC, model_categories(model_id), True)
        for year in VALID_YEARS:
            mask = df["pub_year"].eq(year) & df["has_abstract"].eq(1)
            part = df.loc[mask]
            y = part[label].to_numpy(int)
            p = model.predict(part, embeddings[mask.to_numpy()])
            point, cis = metrics(y, p), bootstrap_metrics(y, p)
            for metric, estimate in point.items():
                lo, hi, valid_boot = cis[metric]
                rows.append({"analysis_type": "abstract_present_fitted_model", "model_id": model_id, "predictor_set": "reduced", "training_population": "P_abs_present_train", "validation_year": year, "document_type": "All", "abstract_status": "present", "n": len(part), "denominator_n": len(part), "events": int(y.sum()), "non_events": int(len(y) - y.sum()), "event_rate": y.mean(), "metric": metric, "estimate": estimate, "ci_lower": lo, "ci_upper": hi, "bootstrap_requested": BOOTSTRAPS, "bootstrap_valid": valid_boot, "notes": "model fitted only among abstract-present training records"})
    return rows


def fit_association(base: pd.DataFrame, include_abstract: bool, original_only: bool = False, interaction: bool = False) -> tuple[pd.DataFrame, sm.discrete.discrete_model.BinaryResultsWrapper, dict, list[str], dict]:
    data = base.copy()
    if original_only:
        data = data[data["is_review"].eq(0)].copy()
    continuous = [c for c in TABLE2_CONTINUOUS if include_abstract or c != "abstract_n_words"]
    binary = [c for c in TABLE2_BINARY if not (original_only and c == "is_review")]
    required = [*continuous, *binary, "is_top10"]
    missing_counts = {column: int(data[column].isna().sum()) for column in required[:-1]}
    estimation = data.dropna(subset=required).copy()
    x = pd.DataFrame(index=estimation.index)
    scales: dict[str, tuple[float, float, str]] = {}
    for column in continuous:
        raw = np.log1p(estimation[column]) if column in AUTHOR_SAFE else estimation[column].astype(float)
        mean, sd = float(raw.mean()), float(raw.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            raise RuntimeError(f"non-estimable standardization for {column}")
        x[column] = (raw - mean) / sd
        scales[column] = (mean, sd, "log1p then z-score" if column in AUTHOR_SAFE else "z-score")
    for column in binary:
        x[column] = estimation[column].astype(float)
    if interaction:
        x["is_review_x_n_references"] = x["n_references"] * x["is_review"]
    design = sm.add_constant(x, has_constant="add")
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise RuntimeError("Table 2 design matrix is rank deficient")
    result = sm.Logit(estimation["is_top10"].astype(int), design).fit(disp=0, maxiter=200)
    ci = result.conf_int()
    if not bool(result.mle_retvals.get("converged")) or not np.isfinite(result.params).all() or not np.isfinite(ci.to_numpy()).all():
        raise RuntimeError("Table 2 coefficient or confidence interval is not finite")
    return estimation, result, scales, binary, missing_counts


def table2_rows(df: pd.DataFrame) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    train = df[df["pub_year"].isin(TRAIN_YEARS)].copy()
    complete_auth = train.dropna(subset=AUTHOR_LOG)
    correlation = complete_auth[AUTHOR_LOG].corr()
    for i, left in enumerate(AUTHOR_SAFE):
        for j, right in enumerate(AUTHOR_SAFE):
            if j <= i:
                continue
            rows.append({"analysis_id": "author_collinearity", "row_type": "collinearity", "population": "P_train complete for four prominence measures", "model_variant": "pre-fit audit", "outcome": "is_top10", "term": f"{left} vs {right}", "term_role": "author prominence", "transformation": "log1p", "n": len(complete_auth), "estimate_type": "pearson_r", "estimate": correlation.loc[f"{left}_log1p", f"{right}_log1p"], "converged": "NA", "separation_or_rank_issue": "none", "notes": "pairwise collinearity audit prespecified before fitting"})
    fitted_results: dict = {}
    for analysis_id, population, include_abstract in [
        ("table2_main_abstract_present", "P_table2_abstract", True),
        ("table2_no_abstract_length", "P_table2_no_abstract_length", False),
    ]:
        estimation, result, scales, binary, missing = fit_association(train, include_abstract)
        fitted_results[analysis_id] = (estimation, result, scales)
        n, events = len(estimation), int(estimation["is_top10"].sum())
        common = {"analysis_id": analysis_id, "population": population, "model_variant": "main" if include_abstract else "missingness sensitivity", "outcome": "is_top10", "n": n, "events": events, "non_events": n - events, "top10_group_n": events, "other_group_n": n - events, "converged": True, "separation_or_rank_issue": "none"}
        rows.append({**common, "row_type": "model_summary", "term": "pseudo_r2", "term_role": "model", "estimate_type": "McFadden pseudo R2", "estimate": result.prsquared, "missing_excluded_n": len(train) - n, "notes": "complete-case maximum-likelihood logistic regression"})
        ci = result.conf_int()
        for term in [*scales, *binary]:
            rows.append({**common, "row_type": "coefficient", "term": term, "term_role": "author prominence" if term in AUTHOR_SAFE else "team size" if term == "n_authors" else "association covariate", "raw_scale": "count" if term in TABLE2_CONTINUOUS else "0/1", "transformation": scales[term][2] if term in scales else "none", "standardization_mean": scales[term][0] if term in scales else np.nan, "standardization_sd": scales[term][1] if term in scales else np.nan, "estimate_type": "adjusted odds ratio", "estimate": math.exp(result.params[term]), "ci_lower": math.exp(ci.loc[term, 0]), "ci_upper": math.exp(ci.loc[term, 1]), "p_value": result.pvalues[term], "notes": "two-sided descriptive p-value"})
        for term, (mean, sd, transformation) in scales.items():
            rows.append({**common, "row_type": "standardization", "term": term, "term_role": "continuous covariate", "raw_scale": "count", "transformation": transformation, "standardization_mean": mean, "standardization_sd": sd, "estimate_type": "raw-unit SD", "estimate": sd, "notes": "estimation-population standardization"})
        for term, count in missing.items():
            rows.append({**common, "row_type": "missingness", "term": term, "term_role": "covariate", "missing_excluded_n": count, "estimate_type": "missing n", "estimate": count, "notes": "per-variable missingness before complete-case restriction; counts may overlap"})
    return rows, fitted_results


def threshold_metric_values(y: np.ndarray, pred: np.ndarray) -> dict[str, tuple[float, int, int]]:
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "sensitivity": (tp / (tp + fn) if tp + fn else np.nan, tp, tp + fn),
        "specificity": (tn / (tn + fp) if tn + fp else np.nan, tn, tn + fp),
        "positive_predictive_value": (tp / (tp + fp) if tp + fp else np.nan, tp, tp + fp),
        "negative_predictive_value": (tn / (tn + fn) if tn + fn else np.nan, tn, tn + fn),
        "manuscripts_flagged_per_100": ((tp + fp) * 100 / len(y) if len(y) else np.nan, tp + fp, len(y)),
    }


def bootstrap_threshold(y: np.ndarray, pred: np.ndarray) -> dict[str, tuple[float, float, int]]:
    rng = np.random.default_rng(SEED)
    values = {name: [] for name in threshold_metric_values(y, pred)}
    for _ in range(BOOTSTRAPS):
        idx = rng.integers(0, len(y), len(y))
        for name, (estimate, _, _) in threshold_metric_values(y[idx], pred[idx]).items():
            if np.isfinite(estimate):
                values[name].append(estimate)
    return {name: (*np.percentile(items, [2.5, 97.5]), len(items)) if items else (np.nan, np.nan, 0) for name, items in values.items()}


def threshold_rows(df: pd.DataFrame, embeddings: np.ndarray, fitted: dict) -> list[dict]:
    rows: list[dict] = []
    train_mask = df["pub_year"].isin(TRAIN_YEARS).to_numpy()
    train = df.loc[train_mask].reset_index(drop=True)
    emb_train = embeddings[train_mask]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    for model_id, label in [("B", "is_top25_jrnl"), ("A", "is_top10")]:
        y_train = train[label].to_numpy(int)
        oof = np.full(len(train), np.nan)
        for fit_index, test_index in cv.split(train, y_train):
            fold = fit_locked(train.iloc[fit_index], emb_train[fit_index], y_train[fit_index], REDUCED_NUMERIC, model_categories(model_id), True)
            oof[test_index] = fold.predict(train.iloc[test_index], emb_train[test_index])
        if np.isnan(oof).any():
            raise RuntimeError("training OOF threshold predictions are incomplete")
        for target in (10, 25):
            cutoff = float(np.quantile(oof, 1 - target / 100, method="higher"))
            actual_training = float((oof >= cutoff).mean() * 100)
            for year in VALID_YEARS:
                mask = df["pub_year"].eq(year).to_numpy()
                part = df.loc[mask]
                y = part[label].to_numpy(int)
                p = fitted[(model_id, "reduced")].predict(part, embeddings[mask])
                pred = (p >= cutoff).astype(int)
                point, cis = threshold_metric_values(y, pred), bootstrap_threshold(y, pred)
                for metric, (estimate, numerator, denominator) in point.items():
                    lo, hi, valid_boot = cis[metric]
                    rows.append({"model_id": model_id, "predictor_set": "reduced", "cv_scheme": "3-fold stratified shuffled OOF; all preprocessing refit in outer training fold", "operating_point": f"flag_{target}_per_100", "target_flagged_per_100": target, "training_probability_cutoff": cutoff, "training_actual_flagged_per_100": actual_training, "validation_year": year, "n": len(part), "events": int(y.sum()), "non_events": int(len(y) - y.sum()), "metric": metric, "numerator": numerator, "denominator": denominator, "estimate": estimate, "ci_lower": lo, "ci_upper": hi, "bootstrap_requested": BOOTSTRAPS, "bootstrap_valid": valid_boot, "random_seed": SEED, "notes": "fixed training-OOF cutoff; all probability ties retained; operating characteristic only"})
    return rows


def ablation_rows(df: pd.DataFrame, embeddings: np.ndarray, fitted: dict, predictions: dict) -> list[dict]:
    rows: list[dict] = []
    train_mask = df["pub_year"].isin(TRAIN_YEARS).to_numpy()
    valid_mask = df["pub_year"].eq(2022).to_numpy()
    train, valid = df.loc[train_mask], df.loc[valid_mask]
    emb_train, emb_valid = embeddings[train_mask], embeddings[valid_mask]
    base = ["n_authors", "n_institutions", "n_countries", "is_international", "is_review", "title_n_words", "abstract_n_words", "has_abstract", "flesch"]
    steps = [
        (1, "team/collaboration + document/text structure", base, False),
        (2, "step 1 + leakage-safe author prominence", [*base, *AUTHOR_LOG], False),
        (3, "step 2 + title/abstract embedding PCs", [*base, *AUTHOR_LOG], True),
        (4, "step 3 + text-derived study-design flags", [*base, *AUTHOR_LOG, "is_rct", "is_sr_meta", "is_cohort", "is_case_report"], True),
        (5, "step 4 + reference count", REDUCED_NUMERIC, True),
    ]
    for model_id, label in [("B", "is_top25_jrnl"), ("A", "is_top10")]:
        y = valid[label].to_numpy(int)
        prior = None
        table3_auc = metrics(y, predictions[(model_id, "reduced", 2022)])["roc_auc"]
        for step, groups, numeric, use_embedding in steps:
            if step == 5:
                p = predictions[(model_id, "reduced", 2022)]
            else:
                model = fit_locked(train, emb_train, train[label].to_numpy(int), numeric, model_categories(model_id), use_embedding)
                p = model.predict(valid, emb_valid)
            auc = roc_auc_score(y, p)
            lo, hi, _ = bootstrap_metrics(y, p)["roc_auc"]
            consistency = "PASS" if step < 5 or auc == table3_auc else "FAIL"
            rows.append({"model_id": model_id, "predictor_set": "reduced", "algorithm": "histogram gradient boosting", "calibration_method": "isotonic, 3-fold training CV", "training_years": "2018-2021", "validation_year": 2022, "feature_step": step, "feature_groups_included": groups, "n": len(valid), "events": int(y.sum()), "non_events": int(len(y) - y.sum()), "roc_auc": auc, "ci_lower": lo, "ci_upper": hi, "delta_from_prior_step": np.nan if prior is None else auc - prior, "table3_comparator_key": f"{model_id}|reduced|2022|roc_auc", "table3_roc_auc": table3_auc, "consistency_status": consistency, "notes": "final step reuses the exact locked reduced-model predictions reported in rev1_model_performance.csv"})
            prior = auc
        if rows[-1]["consistency_status"] != "PASS":
            raise RuntimeError("Figure 2 final bar does not equal the Table 3 reduced-model value")
    return rows


def descriptive_rows(df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []

    def summary(analysis_type: str, grouping: str, level, values: pd.Series, variable: str, document_type: str = "", year="") -> None:
        clean = values.dropna()
        rows.append({"analysis_type": analysis_type, "grouping_variable": grouping, "group_level": level, "document_type": document_type, "publication_year": year, "variable": variable, "statistic": "median_iqr", "n": len(clean), "estimate": clean.median(), "q1": clean.quantile(0.25), "q3": clean.quantile(0.75), "estimable": True})

    types = [("Original article", df["is_review"].eq(0)), ("Review", df["is_review"].eq(1))]
    for label, mask in types:
        summary("citation_by_document_type", "document_type", label, df.loc[mask, "cited_by_count"], "cited_by_count", label)
    u = mannwhitneyu(df.loc[df["is_review"].eq(0), "cited_by_count"], df.loc[df["is_review"].eq(1), "cited_by_count"], alternative="two-sided")
    rows.append({"analysis_type": "citation_by_document_type_test", "grouping_variable": "document_type", "group_level": "overall_test", "variable": "cited_by_count", "statistic": "test", "n": len(df), "test_name": "Mann-Whitney U", "test_statistic": u.statistic, "p_value": u.pvalue, "estimable": True})
    for year in sorted(df["pub_year"].unique()):
        summary("citation_by_publication_year", "publication_year", year, df.loc[df["pub_year"].eq(year), "cited_by_count"], "cited_by_count", year=year)
    kw = kruskal(*[df.loc[df["pub_year"].eq(year), "cited_by_count"] for year in sorted(df["pub_year"].unique())])
    rows.append({"analysis_type": "citation_by_publication_year_test", "grouping_variable": "publication_year", "group_level": "overall_test", "variable": "cited_by_count", "statistic": "test", "n": len(df), "test_name": "Kruskal-Wallis", "test_statistic": kw.statistic, "p_value": kw.pvalue, "estimable": True})
    for label, mask in types:
        summary("reference_count_by_document_type", "document_type", label, df.loc[mask, "n_references"], "n_references", label)
    u = mannwhitneyu(df.loc[df["is_review"].eq(0), "n_references"], df.loc[df["is_review"].eq(1), "n_references"], alternative="two-sided")
    rows.append({"analysis_type": "reference_count_by_document_type_test", "grouping_variable": "document_type", "group_level": "overall_test", "variable": "n_references", "statistic": "test", "n": len(df), "test_name": "Mann-Whitney U", "test_statistic": u.statistic, "p_value": u.pvalue, "estimable": True})
    abstract_table = pd.crosstab(df["is_review"], df["has_abstract"])
    chi = chi2_contingency(abstract_table)
    for label, mask in types:
        denominator = int(mask.sum())
        present = int((mask & df["has_abstract"].eq(1)).sum())
        rows.append({"analysis_type": "abstract_availability_by_document_type", "grouping_variable": "document_type", "group_level": label, "document_type": label, "variable": "has_abstract", "statistic": "n/N", "n": present, "denominator_n": denominator, "estimate": present, "rate": present / denominator, "estimable": True})
        summary("abstract_length_by_document_type", "document_type", label, df.loc[mask & df["has_abstract"].eq(1), "abstract_n_words"], "abstract_n_words", label)
    rows.append({"analysis_type": "abstract_availability_by_document_type_test", "grouping_variable": "document_type", "group_level": "overall_test", "variable": "has_abstract", "statistic": "test", "n": len(df), "test_name": "chi-square", "test_statistic": chi.statistic, "p_value": chi.pvalue, "estimable": True})
    u = mannwhitneyu(df.loc[df["is_review"].eq(0) & df["has_abstract"].eq(1), "abstract_n_words"], df.loc[df["is_review"].eq(1) & df["has_abstract"].eq(1), "abstract_n_words"], alternative="two-sided")
    rows.append({"analysis_type": "abstract_length_by_document_type_test", "grouping_variable": "document_type", "group_level": "overall_test", "variable": "abstract_n_words", "statistic": "test", "n": int(df["has_abstract"].sum()), "test_name": "Mann-Whitney U", "test_statistic": u.statistic, "p_value": u.pvalue, "estimable": True})

    observed = set(zip(df["journal"], df["pub_year"]))
    for journal in sorted(df["journal"].unique()):
        for year in (*TRAIN_YEARS, *VALID_YEARS):
            if (journal, year) not in observed:
                rows.append({"analysis_type": "missing_journal_year_cell", "grouping_variable": "journal_year", "group_level": f"{journal}|{year}", "publication_year": year, "journal": journal, "variable": "cell_count", "statistic": "count", "n": 0, "estimate": 0, "estimable": True})
    for model_id, label, cutoff in [("B", "is_top25_jrnl", 0.75), ("A", "is_top10", 0.90)]:
        for year in VALID_YEARS:
            part = df[df["pub_year"].eq(year)]
            events = int(part[label].sum())
            rows.append({"analysis_type": "label_prevalence_and_ties", "grouping_variable": "publication_year", "group_level": year, "publication_year": year, "model_id": model_id, "variable": label, "statistic": "positive_count_and_rate", "n": len(part), "denominator_n": len(part), "events": events, "non_events": len(part) - events, "estimate": events, "rate": events / len(part), "tie_method": f"{TIE_METHOD}; cutoff={cutoff}", "estimable": True, "notes": "observed prevalence may differ from nominal because tied citation counts share an average rank"})

    train = df[df["pub_year"].isin(TRAIN_YEARS)].copy()
    estimation, result, _, _, _ = fit_association(train, include_abstract=False, original_only=True)
    ci = result.conf_int()
    rows.append({"analysis_type": "original_article_reference_association", "grouping_variable": "document_type", "group_level": "Original article", "document_type": "Original article", "model_id": "A", "variable": "is_top10", "statistic": "adjusted_odds_ratio", "n": len(estimation), "events": int(estimation["is_top10"].sum()), "non_events": int(len(estimation) - estimation["is_top10"].sum()), "effect_term": "n_references", "effect_estimate": math.exp(result.params["n_references"]), "ci_lower": math.exp(ci.loc["n_references", 0]), "ci_upper": math.exp(ci.loc["n_references", 1]), "p_value": result.pvalues["n_references"], "estimable": True, "notes": "all-training-record sensitivity specification; abstract length and constant review-status term omitted"})

    interaction_note = ""
    estimable = True
    try:
        for review_value in (0, 1):
            if train.loc[train["is_review"].eq(review_value), "is_top10"].nunique() < 2:
                raise RuntimeError("both outcome classes do not occur within each document type")
        estimation, result, _, _, _ = fit_association(train, include_abstract=False, interaction=True)
        ci = result.conf_int()
        term = "is_review_x_n_references"
        effect, lo, hi, p_value = math.exp(result.params[term]), math.exp(ci.loc[term, 0]), math.exp(ci.loc[term, 1]), result.pvalues[term]
    except Exception as exc:
        estimable, interaction_note = False, str(exc)
        estimation, effect, lo, hi, p_value = train.iloc[0:0], np.nan, np.nan, np.nan, np.nan
    rows.append({"analysis_type": "review_status_reference_count_interaction", "grouping_variable": "document_type", "group_level": "interaction", "model_id": "A", "variable": "is_top10", "statistic": "interaction_odds_ratio", "n": len(estimation), "events": int(estimation["is_top10"].sum()) if len(estimation) else np.nan, "non_events": int(len(estimation) - estimation["is_top10"].sum()) if len(estimation) else np.nan, "effect_term": "is_review_x_n_references", "effect_estimate": effect, "ci_lower": lo, "ci_upper": hi, "p_value": p_value, "estimable": estimable, "notes": interaction_note or "prespecified interaction using standardized reference count"})
    return rows


def write_figure(ablation_path: Path, figure_path: Path, manifest: dict) -> None:
    data = pd.read_csv(ablation_path)
    if sha256(ablation_path) == "":
        raise RuntimeError("ablation CSV hash failed")
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colors = {"B": "#D55E00", "A": "#0072B2"}
    hatches = {"B": "//", "A": ""}
    x = np.arange(5)
    width = 0.36
    labels = ["Core\nstructure", "+ Author", "+ Text", "+ Study\ndesign", "+ References"]
    for offset, model_id in [(-width / 2, "B"), (width / 2, "A")]:
        part = data[data["model_id"].eq(model_id)].sort_values("feature_step")
        bars = ax.bar(x + offset, part["roc_auc"], width, label=f"Model {model_id}", color=colors[model_id], edgecolor="black", linewidth=0.6, hatch=hatches[model_id])
        for bar, value in zip(bars, part["roc_auc"]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels)
    low = max(0.5, math.floor((data["roc_auc"].min() - 0.04) * 20) / 20)
    high = min(1.0, math.ceil((data["roc_auc"].max() + 0.04) * 20) / 20)
    ax.set_ylim(low, high)
    ax.set_ylabel("ROC-AUC (2022 held-out)")
    ax.set_title("Cumulative feature-group contribution", loc="left")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=600, metadata={"Title": "REV1 Figure 2 ablation", "Description": f"Generated directly from {ablation_path.name}; freeze_id={manifest['freeze_id']}; plan_sha256={PLAN_SHA256}"})
    plt.close(fig)


def validate_outputs(outputs: dict[str, pd.DataFrame], prov: dict) -> None:
    if set(outputs) != set(OUTPUT_COLUMNS):
        raise RuntimeError("output file set differs from the seven authorized CSVs")
    for name, data in outputs.items():
        if list(data.columns) != OUTPUT_COLUMNS[name] or data.empty:
            raise RuntimeError(f"invalid output schema or empty output: {name}")
        for column, value in prov.items():
            if set(data[column].astype(str)) != {str(value)}:
                raise RuntimeError(f"provenance mismatch in {name}: {column}")
    perf = outputs["rev1_model_performance.csv"]
    ablation = outputs["rev1_ablation.csv"]
    for model_id in ("B", "A"):
        table3 = perf[(perf["model_id"] == model_id) & (perf["predictor_set"] == "reduced") & (perf["validation_year"] == 2022) & (perf["metric"] == "roc_auc")]["estimate"].iloc[0]
        final = ablation[(ablation["model_id"] == model_id) & (ablation["feature_step"] == 5)]["roc_auc"].iloc[0]
        if table3 != final:
            raise RuntimeError(f"Figure 2/Table 3 mismatch for Model {model_id}")


def main() -> None:
    if canonical_plan_sha256(PLAN) != PLAN_SHA256:
        raise RuntimeError("approved canonical analysis-plan hash does not match")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["approved_plan"]["canonical_sha256"] != PLAN_SHA256 or sha256(SCRIPT) != manifest["analysis_script"]["sha256"]:
        raise RuntimeError("manifest plan/script hash does not match")
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    prov = provenance(manifest, generated_at)
    df = load_data(manifest)

    np.random.seed(SEED)
    text = (df["title"].fillna("").str.strip() + np.where(df["has_abstract"].eq(1), ". " + df["abstract"].fillna("").str.strip(), "")).tolist()
    encoder = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device="cpu")
    embeddings = np.asarray(encoder.encode(text, batch_size=64, show_progress_bar=True, normalize_embeddings=True), dtype=np.float32)
    if embeddings.shape != (len(df), 384):
        raise RuntimeError("unexpected all-MiniLM-L6-v2 embedding shape")

    model_rows, fitted, predictions = performance_rows(df, embeddings)
    outputs = {
        "rev1_predictor_availability.csv": frame(predictor_rows(), "rev1_predictor_availability.csv", prov),
        "rev1_model_performance.csv": frame(model_rows, "rev1_model_performance.csv", prov),
        "rev1_abstract_checks.csv": frame(abstract_rows(df, embeddings, fitted), "rev1_abstract_checks.csv", prov),
    }
    table_rows, _ = table2_rows(df)
    outputs["rev1_table2_models.csv"] = frame(table_rows, "rev1_table2_models.csv", prov)
    outputs["rev1_threshold_performance.csv"] = frame(threshold_rows(df, embeddings, fitted), "rev1_threshold_performance.csv", prov)
    outputs["rev1_ablation.csv"] = frame(ablation_rows(df, embeddings, fitted, predictions), "rev1_ablation.csv", prov)
    outputs["rev1_descriptive_checks.csv"] = frame(descriptive_rows(df), "rev1_descriptive_checks.csv", prov)
    validate_outputs(outputs, prov)

    RESULTS.mkdir(exist_ok=True)
    for name, data in outputs.items():
        data.to_csv(RESULTS / name, index=False, lineterminator="\n")
    figure = RESULTS / "rev1_figure2_ablation.png"
    write_figure(RESULTS / "rev1_ablation.csv", figure, manifest)
    if not figure.exists() or figure.stat().st_size == 0:
        raise RuntimeError("Figure 2 was not generated")
    print(json.dumps({"generated_at": generated_at, "csvs": {name: len(data) for name, data in outputs.items()}, "figure": str(figure.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
