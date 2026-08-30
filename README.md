# spine-citation-predictor

Code and cohort identifiers for *Citation-Naive Prediction of Future Impact Among Published Spine Articles: A Bibliometric and Temporal Validation Study*.

The study asks how far the later citation propensity of a published spine article can be predicted from information that approximates what is available before acceptance. Models were trained on articles published in 2018 to 2021 in 13 spine journals and evaluated, unchanged, in the held-out years 2022 and 2023.

## What is here

| Path | Contents |
|---|---|
| `data/work_ids.txt` | The 13,299 OpenAlex work IDs that define the frozen cohort, one per line |
| `scripts/` | Retrieval and feature construction: OpenAlex collection, cohort and outcome labels, author history counted only through the year preceding publication, and text-derived study-design and readability features |
| `data/py/rev1_focused_analysis.py` | The analysis that produced every number in the manuscript: the locked reduced and expanded models, the association models, the abstract-availability analyses, the operating thresholds, the feature-group ablation and the descriptive checks |
| `results/` | The frozen outputs those analyses wrote. Every value in the manuscript, its tables and its supplement can be traced to these files |
| `data/rev1_reproducibility_manifest.json` | Cohort hash, work-ID list hash, package versions, random seeds and the audit of author-feature availability |

## Data

No article records are redistributed here. All bibliographic data come from [OpenAlex](https://openalex.org), which releases its data under CC0, and `data/work_ids.txt` identifies exactly which records were used. Citation counts change as OpenAlex is updated, so a fresh retrieval will not reproduce the frozen values exactly; the analysis was run against a snapshot retrieved in June 2026, whose hash is recorded in the manifest.

The study analysed public bibliographic metadata only. It involved no human participants and no identifiable patient information.

## Reproducing the analysis

```bash
pip install -r requirements.txt
export OPENALEX_MAILTO="you@example.org"      # optional, OpenAlex polite pool
export OPENALEX_API_KEY="..."                 # optional, only if you have one

python scripts/01_collect_openalex.py         # retrieve the works
python scripts/31_collect_1819.py             # append 2018-2019
python scripts/02_features.py                 # cohort, outcomes, base features
python scripts/07_author_features.py          # leakage-safe author history
python scripts/13_features_v3.py              # text-derived features
python data/py/rev1_focused_analysis.py       # models, associations, thresholds, ablation
```

Each script resolves the project root from its own location; set `CITATION_PREDICTOR_ROOT` to override. Text embeddings use `sentence-transformers/all-MiniLM-L6-v2`; the revision pinned in the manifest is the one that was used. The random seed is 42 throughout.

## Two things worth knowing before reusing this

The cohort contains only articles that were ultimately published. An editorial submission pool also contains the manuscripts a journal will reject, and the models were neither trained nor tested on such a pool, so the reported performance need not carry over to it.

The predictors are publication-record proxies, not the submitted versions. Titles, abstracts, author lists and reference lists can change during peer review. The models estimate citability, meaning a set of machine-legible correlates of citation, and not scientific merit or clinical value.

## Citation

Park J, Park SM, Kim HJ, Yeom JS. Citation-Naive Prediction of Future Impact Among Published Spine Articles: A Bibliometric and Temporal Validation Study. *Asian Spine Journal*. Under review.

## License

MIT, see `LICENSE`.
