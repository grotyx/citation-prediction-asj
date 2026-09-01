"""REV1 add-on: restore the three prespecified sensitivity analyses on the locked
reduced predictor set, and export the locked reduced models for the public web app.

Imports rev1_focused_analysis rather than editing it, because that script verifies
its own sha256 against data/rev1_reproducibility_manifest.json; editing it would
invalidate the published reproducibility manifest.

Outputs
  results/rev1_sensitivity_restored.csv   original-only / country-blind / fixed-3y-window
  model_rev1/{model_A,model_B}_reduced.pkl + constants_reduced.json   web-app artifacts
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev1_focused_analysis import (  # noqa: E402
    BOOTSTRAPS,
    DATA,
    MANIFEST,
    MODEL_NAME,
    MODEL_REVISION,
    REDUCED_NUMERIC,
    RESULTS,
    ROOT,
    SEED,
    TRAIN_YEARS,
    VALID_YEARS,
    bootstrap_metrics,
    fit_locked,
    load_data,
    metrics,
    model_categories,
    provenance,
    sha256,
)

MODEL_OUT = ROOT / "model_rev1"
# country-blind drops the two geographic variables that survive into the reduced set
# (has_korea / has_usa were already expanded-only)
COUNTRY_FEATURES = ["n_countries", "is_international"]
COUNTRY_BLIND_NUMERIC = [c for c in REDUCED_NUMERIC if c not in COUNTRY_FEATURES]


def evaluate(model, df, embeddings, label, mask, row, rows):
    valid = df.loc[mask]
    p = model.predict(valid, embeddings[mask])
    y = valid[label].to_numpy(int)
    point, cis = metrics(y, p), bootstrap_metrics(y, p)
    for metric, estimate in point.items():
        lo, hi, valid_boot = cis[metric]
        rows.append(
            {
                **row,
                "n": len(valid),
                "events": int(y.sum()),
                "non_events": int(len(y) - y.sum()),
                "event_rate": y.mean(),
                "metric": metric,
                "estimate": estimate,
                "ci_lower": lo,
                "ci_upper": hi,
                "bootstrap_requested": BOOTSTRAPS,
                "bootstrap_valid": valid_boot,
                "random_seed": SEED,
            }
        )
    return p


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    df = load_data(manifest)
    prov = provenance(manifest, datetime.now().astimezone().isoformat(timespec="seconds"))

    np.random.seed(SEED)
    text = (
        df["title"].fillna("").str.strip()
        + np.where(df["has_abstract"].eq(1), ". " + df["abstract"].fillna("").str.strip(), "")
    ).tolist()
    encoder = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device="cpu")
    embeddings = np.asarray(
        encoder.encode(text, batch_size=64, show_progress_bar=True, normalize_embeddings=True),
        dtype=np.float32,
    )
    if embeddings.shape != (len(df), 384):
        raise RuntimeError("unexpected all-MiniLM-L6-v2 embedding shape")

    # fixed 3-year-window labels, using the same average-rank tie convention as REV1
    df["fw_top25_jrnl"] = (
        df.groupby(["journal", "pub_year"])["c2_early"].rank(method="average", pct=True).ge(0.75).astype(int)
    )
    df["fw_top10"] = df.groupby("pub_year")["c2_early"].rank(method="average", pct=True).ge(0.90).astype(int)

    train_mask = df["pub_year"].isin(TRAIN_YEARS).to_numpy()
    rows: list[dict] = []
    exported: dict = {}

    for model_id, label, fw_label in [("B", "is_top25_jrnl", "fw_top25_jrnl"), ("A", "is_top10", "fw_top10")]:
        categorical = model_categories(model_id)

        # --- S-a. locked reduced set (also the artifact exported to the web app) ---
        locked = fit_locked(
            df.loc[train_mask], embeddings[train_mask], df.loc[train_mask, label].to_numpy(int),
            REDUCED_NUMERIC, categorical, True, False,
        )
        exported[model_id] = locked

        # --- S-b. original articles only (reviews excluded) — R1-C1, submitted Table S4 ---
        orig_train = (train_mask) & df["is_review"].eq(0).to_numpy()
        orig_model = fit_locked(
            df.loc[orig_train], embeddings[orig_train], df.loc[orig_train, label].to_numpy(int),
            REDUCED_NUMERIC, categorical, True, False,
        )
        for year in VALID_YEARS:
            mask = (df["pub_year"].eq(year) & df["is_review"].eq(0)).to_numpy()
            evaluate(orig_model, df, embeddings, label, mask,
                     {**prov, "model_id": model_id, "sensitivity": "original_articles_only",
                      "predictor_set": "reduced", "label_definition": "cumulative citations",
                      "training_population": "2018-2021 original articles",
                      "validation_year": year, "population": f"P_valid_{year}_original_articles",
                      "notes": "reviews excluded from training and validation; restores submitted Supplementary Table S4 on the locked reduced set"},
                     rows)

        # --- S-c. country-blind (drops n_countries, is_international) ---
        cb_model = fit_locked(
            df.loc[train_mask], embeddings[train_mask], df.loc[train_mask, label].to_numpy(int),
            COUNTRY_BLIND_NUMERIC, categorical, True, False,
        )
        for year in VALID_YEARS:
            mask = df["pub_year"].eq(year).to_numpy()
            evaluate(cb_model, df, embeddings, label, mask,
                     {**prov, "model_id": model_id, "sensitivity": "country_blind",
                      "predictor_set": "reduced_without_geography",
                      "label_definition": "cumulative citations",
                      "training_population": "2018-2021", "validation_year": year,
                      "population": f"P_valid_{year}",
                      "notes": "n_countries and is_international removed; has_korea/has_usa were already expanded-only"},
                     rows)

        # --- S-d. fixed 3-year citation window labels ---
        fw_model = fit_locked(
            df.loc[train_mask], embeddings[train_mask], df.loc[train_mask, fw_label].to_numpy(int),
            REDUCED_NUMERIC, categorical, True, False,
        )
        for year in VALID_YEARS:
            mask = df["pub_year"].eq(year).to_numpy()
            evaluate(fw_model, df, embeddings, fw_label, mask,
                     {**prov, "model_id": model_id, "sensitivity": "fixed_3_year_window",
                      "predictor_set": "reduced",
                      "label_definition": "citations within 3 years of publication (c2_early)",
                      "training_population": "2018-2021", "validation_year": year,
                      "population": f"P_valid_{year}",
                      "notes": "outcome relabelled on a fixed window; same average-rank tie convention"},
                     rows)

        # label agreement between cumulative and fixed-window definitions
        for year in VALID_YEARS:
            sub = df[df["pub_year"].eq(year)]
            rows.append({**prov, "model_id": model_id, "sensitivity": "fixed_3_year_window",
                         "predictor_set": "label_agreement",
                         "label_definition": "cumulative vs fixed 3-year window",
                         "training_population": "", "validation_year": year,
                         "population": f"P_valid_{year}", "n": len(sub),
                         "events": int(sub[label].sum()), "non_events": int(len(sub) - sub[label].sum()),
                         "event_rate": float(sub[label].mean()), "metric": "label_agreement",
                         "estimate": float((sub[label].astype(int) == sub[fw_label]).mean()),
                         "ci_lower": np.nan, "ci_upper": np.nan, "bootstrap_requested": 0,
                         "bootstrap_valid": 0, "random_seed": SEED,
                         "notes": "proportion of articles receiving the same label under both definitions"})

    out = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / "rev1_sensitivity_restored.csv", index=False, encoding="utf-8")

    # ---- web-app export: locked reduced models + corpus constants (reduced set only) ----
    MODEL_OUT.mkdir(exist_ok=True)
    train = df.loc[train_mask]
    for model_id, model in exported.items():
        joblib.dump(
            {"numeric": model.numeric, "categorical": model.categorical,
             "pca": model.pca, "category_maps": model.category_maps,
             "classifier": model.classifier},
            MODEL_OUT / f"model_{model_id}_reduced.pkl",
        )
    constants = {
        "predictor_set": "reduced pre-acceptance analog (REV1 locked)",
        "training_years": list(TRAIN_YEARS),
        "embedding_model": MODEL_NAME,
        "embedding_revision": MODEL_REVISION,
        "numeric_features": REDUCED_NUMERIC,
        "medians": {
            c: (None if train[c].dropna().empty else float(train[c].median()))
            for c in ["n_authors", "n_institutions", "n_countries", "is_international",
                      "title_n_words", "abstract_n_words", "flesch", "n_references",
                      "fa_prior_works", "fa_prior_cites", "la_prior_works", "la_prior_cites"]
        },
        "event_rate": {
            "model_B_within_journal_top25": float(train["is_top25_jrnl"].mean()),
            "model_A_field_top10": float(train["is_top10"].mean()),
        },
        "source_cohort_sha256": sha256(DATA / "features.parquet"),
        "generated_at": prov["generated_at"],
    }
    (MODEL_OUT / "constants_reduced.json").write_text(
        json.dumps(constants, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"rows": len(out), "models": sorted(exported), "out": str(MODEL_OUT)}, indent=2))


if __name__ == "__main__":
    main()
