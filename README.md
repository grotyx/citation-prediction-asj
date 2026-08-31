# Spine Citation Predictor 📈

🔗 **Live app:** https://asj-citation.streamlit.app/

A research-stage web app that estimates the **later citation propensity of a spine article** from
publication-record variables (title, abstract, reference count, article type). It is the deployable
companion to the study *"Citation-Naive Machine-Learning Prediction of Future Impact Among Published
Spine Articles: A Bibliometric and Temporal Validation Study."*

> ⚠️ **Research adjunct only.** This tool supports — but does not replace — scientific peer review.
> It must never be used as a determinant of acceptance or rejection.

> ⚠️ **What the score is not.** The models estimate *citability* — a set of surface-level,
> machine-legible correlates of citation — not scientific merit, methodological rigor, or clinical
> value. The associations behind it are observational and are not actionable strategies for authors.

## Scope and limits of the evidence

- The models were developed and validated on **articles that were ultimately published**. Original
  submitted versions and rejected manuscripts were unavailable, so performance in a real editorial
  submission pool is **untested, in either direction**.
- The predictors are **publication-record proxies** for information conceptually available before
  acceptance, not the submitted manuscript itself. Titles, abstracts, author lists and reference
  lists may change during peer review.
- About one-third of corpus records lacked a machine-readable abstract, so part of the training
  signal comes from title-only inputs, which no real submission presents.

## What it reports

For a pasted title/abstract (or an uploaded manuscript PDF):

| Output | Meaning |
|---|---|
| **Within-journal top-25% probability** | Probability the article lands in the top 25% of citations within its journal and year (Model B, the study's primary model). Baseline = 25%. |
| **Field top-10% probability** | Probability of top 10% across all 13 spine journals (Model A, secondary). Model A also sees journal identity, which is why it discriminates better. |
| **Score drivers** | Which inputs raise or lower the score. These are model sensitivities, not causal effects. |

The review/meta-analysis flag is suggested automatically — from a keyword match on the title/abstract in the manual-entry tab, or from the PDF's `Manuscript Type`/`Article Type` field in the upload tab — and can be overridden by hand.

## Model

- Corpus: 13 dedicated spine journals, **2018–2023** (n = 13,299), from OpenAlex.
- Training **2018–2021**, temporal validation in **2022 and 2023** separately.
- No citation measure of the index article enters either model; all preprocessing is fit on training
  years only.
- Algorithm: histogram gradient boosting with isotonic calibration; text via Sentence-BERT
  (`all-MiniLM-L6-v2`) → 50-component PCA.
- Reported performance (reduced pre-acceptance-analog predictor set, primary Model B):
  ROC-AUC 0.721 (95% CI 0.695–0.746) in 2022 and 0.706 (0.683–0.729) in 2023.
- At a threshold flagging ~14 articles per 100, sensitivity is ~0.30 — the score misses roughly
  two-thirds of eventual top-quartile articles.

Analysis code, the OpenAlex work-ID list, the frozen result files and the reproducibility manifest
for the study are at **https://github.com/grotyx/spine-citation-predictor**.

### Artifact status

The bundled models in `models/` are the **locked reduced pre-acceptance-analog models** reported as
primary in the revised study: `model_B_reduced.pkl` (within-journal top quartile),
`model_A_reduced.pkl` (field top decile) and `constants_reduced.json` (training-cohort medians,
event rates, and the pinned embedding revision). Variables that OpenAlex records only after
publication — topic and subfield annotations and their derivatives, open-access status, the current
last-author h-index — are **not** used. The full corpus is not required at inference.

The 3-year expected-citation regressor that earlier versions of this app displayed has been removed:
it was never reported or validated in the study.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (default http://localhost:8501). The first run downloads the
`all-MiniLM-L6-v2` model (~80 MB) from Hugging Face.

## Deploy

Works on **Streamlit Community Cloud** or **Hugging Face Spaces** (Streamlit SDK): point it at this
repo with `app.py` as the entry point. No GPU or secrets required.

## Files

```
app.py             # Streamlit UI
scorer.py          # self-contained scoring (Model A/B + drivers); `python scorer.py` runs a self-check
pdf_extract.py     # title/abstract/reference-count extraction from a PDF
models/            # locked reduced models + constants_reduced.json
requirements.txt
```

## Limitations

Citation-naive prediction has an intrinsic ceiling; citation also depends on post-publication
factors (topic timeliness, promotion, and chance). Citation labels come from OpenAlex, which differs
in absolute counts from subscription indices. Preferentially promoting high-scored articles could
reinforce cumulative advantage and create a self-fulfilling citation process. Some inputs could
encode geographic or other bias and must not substitute for content review.

## Author

**Professor Sang-Min Park, M.D., Ph.D.**
Department of Orthopaedic Surgery, Seoul National University Bundang Hospital,
Seoul National University College of Medicine
🌐 [sangmin.me](https://sangmin.me/)

## License

BSD 3-Clause — see [LICENSE](LICENSE).
