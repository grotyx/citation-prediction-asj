# Spine Citation Predictor 📈

🔗 **Live app:** https://asj-citation.streamlit.app/

Code, cohort identifiers and deployable app for *Citation-Naive Machine-Learning Prediction of
Future Impact Among Published Spine Articles: A Bibliometric and Temporal Validation Study*.

The study asks how far the later citation propensity of a published spine article can be predicted
from information that approximates what is available before acceptance. Models were trained on
articles published in 2018 to 2021 in 13 spine journals and evaluated, unchanged, in the held-out
years 2022 and 2023.

> ⚠️ **Research adjunct only.** The app supports — but does not replace — scientific peer review.
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

## What is here

| Path | Contents |
|---|---|
| `app.py`, `scorer.py`, `pdf_extract.py`, `models/` | The web app and the locked reduced Model A and Model B it serves |
| `data/work_ids.txt` | The 13,299 OpenAlex work IDs that define the frozen cohort, one per line |
| `scripts/` | Retrieval and feature construction: OpenAlex collection, cohort and outcome labels, author history counted only through the year preceding publication, and text-derived study-design and readability features |
| `data/py/rev1_focused_analysis.py` | The analysis that produced every number in the manuscript: the locked reduced and expanded models, the association models, the abstract-availability analyses, the operating thresholds, the feature-group ablation and the descriptive checks |
| `data/py/rev1_sensitivity_and_export.py` | The three sensitivity analyses prespecified in the submitted manuscript, repeated on the locked reduced model: original articles only, a country-blind model, and fixed 3-year-window labels. It also writes the model artifacts the app serves. It imports the analysis above rather than editing it, because that script verifies its own hash against the manifest |
| `results/` | The frozen outputs those analyses wrote. Every value in the manuscript, its tables and its supplement can be traced to these files |
| `data/rev1_reproducibility_manifest.json` | Cohort hash, work-ID list hash, package versions, random seeds and the audit of author-feature availability |

The app and the analysis live in one repository on purpose. They were split once, and the app went
on advertising submission-time, leakage-safe prediction after the revision had retracted that claim,
while serving models fitted on the superseded predictor set. Keeping them together ties the models
the app serves to the script that writes them.

## The app

For a pasted title/abstract, or an uploaded manuscript PDF:

| Output | Meaning |
|---|---|
| **Within-journal top-25% probability** | Probability the article lands in the top 25% of citations within its journal and year (Model B, the study's primary model). |
| **Field top-10% probability** | Probability of top 10% across all 13 spine journals (Model A, secondary). Model A also sees journal identity, which is why it discriminates better. |
| **Score drivers** | Which inputs raise or lower the score. These are model sensitivities, not causal effects. |

The review/meta-analysis flag is suggested automatically — from a keyword match on the
title/abstract in the manual-entry tab, or from the PDF's `Manuscript Type`/`Article Type` field in
the upload tab — and can be overridden by hand.

```bash
pip install -r requirements.txt
streamlit run app.py
python scorer.py          # self-check
```

The first run downloads `all-MiniLM-L6-v2` (~80 MB) from Hugging Face. The bundled models in
`models/` are written by `data/py/rev1_sensitivity_and_export.py`; the scikit-learn pin in
`requirements.txt` has to match the version that fitted them.

## Model

- Corpus: 13 dedicated spine journals, **2018–2023** (n = 13,299), from OpenAlex.
- Training **2018–2021**, temporal validation in **2022 and 2023** separately.
- No citation measure of the index article enters either model; all preprocessing is fit on training
  years only. Variables OpenAlex records only after publication — topic and subfield annotations and
  their derivatives, open-access status, the current last-author h-index — are excluded.
- Algorithm: histogram gradient boosting with isotonic calibration; text via Sentence-BERT
  (`all-MiniLM-L6-v2`) → 50-component PCA.
- Reported performance (reduced pre-acceptance-analog predictor set, primary Model B):
  ROC-AUC 0.721 (95% CI 0.695–0.746) in 2022 and 0.706 (0.683–0.729) in 2023.
- At a threshold flagging ~14 articles per 100, sensitivity is ~0.30 — the score misses roughly
  two-thirds of eventual top-quartile articles.

## Data

No article records are redistributed here. All bibliographic data come from
[OpenAlex](https://openalex.org), which releases its data under CC0, and `data/work_ids.txt`
identifies exactly which records were used. Citation counts change as OpenAlex is updated, so a
fresh retrieval will not reproduce the frozen values exactly; the analysis was run against a
snapshot retrieved in June 2026, whose hash is recorded in the manifest.

The study analysed public bibliographic metadata only. It involved no human participants and no
identifiable patient information.

## Reproducing the analysis

```bash
pip install -r requirements-analysis.txt
export OPENALEX_MAILTO="you@example.org"      # optional, OpenAlex polite pool
export OPENALEX_API_KEY="..."                 # optional, only if you have one

python scripts/01_collect_openalex.py         # retrieve the works
python scripts/31_collect_1819.py             # append 2018-2019
python scripts/02_features.py                 # cohort, outcomes, base features
python scripts/07_author_features.py          # leakage-safe author history
python scripts/13_features_v3.py              # text-derived features
python data/py/rev1_focused_analysis.py       # models, associations, thresholds, ablation
python data/py/rev1_sensitivity_and_export.py # prespecified sensitivity analyses, app artifacts
```

`requirements-analysis.txt` holds the pinned analysis stack; the root `requirements.txt` is the
lighter set the deployed app needs. Each script resolves the project root from its own location; set
`CITATION_PREDICTOR_ROOT` to override. Text embeddings use `sentence-transformers/all-MiniLM-L6-v2`;
the revision pinned in the manifest is the one that was used. The random seed is 42 throughout.

## Two things worth knowing before reusing this

The cohort contains only articles that were ultimately published. An editorial submission pool also
contains the manuscripts a journal will reject, and the models were neither trained nor tested on
such a pool, so the reported performance need not carry over to it.

The predictors are publication-record proxies, not the submitted versions. Titles, abstracts, author
lists and reference lists can change during peer review. The models estimate citability, meaning a
set of machine-legible correlates of citation, and not scientific merit or clinical value.

## Limitations

Citation-naive prediction has an intrinsic ceiling; citation also depends on post-publication
factors (topic timeliness, promotion, and chance). Citation labels come from OpenAlex, which differs
in absolute counts from subscription indices. Preferentially promoting high-scored articles could
reinforce cumulative advantage and create a self-fulfilling citation process. Some inputs could
encode geographic or other bias and must not substitute for content review.

## Citation

Park J, Park SM, Kim HJ, Yeom JS. Citation-Naive Machine-Learning Prediction of Future Impact Among
Published Spine Articles: A Bibliometric and Temporal Validation Study. *Asian Spine Journal*. Under
review.

## Author

**Professor Sang-Min Park, M.D., Ph.D.**
Department of Orthopaedic Surgery, Seoul National University Bundang Hospital,
Seoul National University College of Medicine
🌐 [sangmin.me](https://sangmin.me/)

## License

BSD-3-Clause, see `LICENSE`.
