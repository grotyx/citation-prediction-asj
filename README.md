# Spine Citation Predictor 📈

A web app that estimates the **future citation impact of a spine manuscript at the time of submission**, using only information available before publication (title, abstract, reference count, article type). It is the deployable companion to the study *"Predicting Citation Impact of Spine Research at the Time of Submission."*

> ⚠️ **Auxiliary triage aid only.** This tool supports — but does not replace — scientific peer review. It must never be used as a determinant of acceptance or rejection.

## What it reports

For a pasted title/abstract (or an uploaded manuscript PDF):

| Output | Meaning |
|---|---|
| **ASJ top-25% probability** | Probability the article lands in the top 25% of citations *within Asian Spine Journal* for its year (Model B, the primary model). Baseline = 25%. |
| **3-year expected citations** | Predicted citations ~3 years after publication, with a rough range and comparison to the ASJ / field average. |
| **Field top-10% probability** | Probability of top 10% across all 13 spine journals (Model A, reference only). |
| **Score drivers** | Which factors raise/lower the score (reference count, review/meta-analysis, topic novelty, abstract length). |

## Model

- Corpus: 13 dedicated spine journals, **2018–2023** (n = 13,299), from OpenAlex.
- Training **2018–2021**, temporal validation **2022 and 2023**.
- Leakage-safe: only submission-time predictors; no citation-derived variables; all preprocessing fit on training years only.
- Algorithm: histogram gradient boosting with isotonic calibration; text via Sentence-BERT (`all-MiniLM-L6-v2`) → PCA.
- Performance: within-journal Model B ROC-AUC ≈ 0.72 (stable across 2022/2023); ASJ subgroup Spearman ρ ≈ 0.58.

Bundled artifacts (`models/`): `model_B_final.pkl`, `model_A_final.pkl`, `reg_c3_final.pkl`, `text_artifacts_final.pkl`, and `constants.json` (corpus medians + ASJ anchors). The full corpus is **not** required at inference.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (default http://localhost:8501). The first run downloads the `all-MiniLM-L6-v2` model (~80 MB) from Hugging Face.

## Deploy

Works on **Streamlit Community Cloud** or **Hugging Face Spaces** (Streamlit SDK): point it at this repo with `app.py` as the entry point. No GPU or secrets required.

## Files

```
app.py             # Streamlit UI
scorer.py          # self-contained scoring (Model A/B + 3-yr regressor + drivers)
pdf_extract.py     # title/abstract/reference-count extraction from a PDF
models/            # trained models + constants.json
requirements.txt
```

## Limitations

Prediction at submission has an intrinsic ceiling; citation also depends on post-publication factors (timeliness, promotion, chance). Citation labels come from OpenAlex, which differs in absolute counts from subscription indices. Some inputs (e.g., country) could encode bias and must not substitute for content review.
