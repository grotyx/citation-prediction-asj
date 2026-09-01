# -*- coding: utf-8 -*-
"""
v3 추가 피처 (API 불필요 — 기존 raw_works.jsonl + features.parquet 로 파생).
  - 연구설계 플래그: is_rct, is_sr_meta, is_cohort, is_case_report (초록 정규식)
  - 학제성: n_topics (OpenAlex topics 개수)
  - 신호어: sw_novel, sw_ai, sw_guideline (제목+초록)
  - 가독성: flesch (textstat 있으면, 없으면 근사식)
  - 주제 hotness: subfield_size, subfield_year_count, subfield_growth (출판수 기반, 누수 없음)
출력: data/features_v3extra.parquet (key=id)
"""
import os, json, re, math
import pandas as pd

ROOT = os.environ.get("CITATION_PREDICTOR_ROOT",
                     os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")

feat = pd.read_parquet(os.path.join(DATA, "features.parquet"))  # id, pub_year, topic_subfield, title, abstract...
keep_ids = set(feat["id"])

# raw 에서 topics 개수 재파생
def reconstruct_abstract(inv):
    if not inv: return ""
    pos = {}
    for w, idxs in inv.items():
        for i in idxs: pos[i] = w
    return " ".join(pos[i] for i in sorted(pos))

rawmap = {}
with open(os.path.join(DATA, "raw_works.jsonl"), encoding="utf-8") as fp:
    for line in fp:
        w = json.loads(line)
        wid = (w.get("id") or "").split("/")[-1]
        if wid in keep_ids and wid not in rawmap:
            rawmap[wid] = {"n_topics": len(w.get("topics") or [])}

try:
    import textstat
    def flesch(t):
        return textstat.flesch_reading_ease(t) if t else 0.0
    HAVE_TS = True
except Exception:
    HAVE_TS = False
    def _syll(word):
        word = word.lower(); v = "aeiouy"; n = 0; prev = False
        for c in word:
            isv = c in v
            if isv and not prev: n += 1
            prev = isv
        if word.endswith("e"): n = max(1, n-1)
        return max(1, n)
    def flesch(t):
        if not t: return 0.0
        sents = max(1, len(re.findall(r"[.!?]+", t)))
        words = re.findall(r"[A-Za-z]+", t)
        if not words: return 0.0
        syll = sum(_syll(w) for w in words)
        W = len(words)
        return 206.835 - 1.015*(W/sents) - 84.6*(syll/W)

RX = {
    "is_rct": re.compile(r"\brandomi[sz]ed|randomi[sz]ed controlled|\bRCT\b", re.I),
    "is_sr_meta": re.compile(r"\bmeta-?analysis|systematic review\b", re.I),
    "is_cohort": re.compile(r"\bcohort\b|prospective|retrospective", re.I),
    "is_case_report": re.compile(r"\bcase report|case series\b", re.I),
    "sw_novel": re.compile(r"\bnovel\b|first[- ]time|for the first time", re.I),
    "sw_ai": re.compile(r"machine learning|deep learning|artificial intelligence|\bAI\b|radiomic|neural network", re.I),
    "sw_guideline": re.compile(r"guideline|consensus|recommendation", re.I),
}

rows = []
for _, r in feat.iterrows():
    text = ((r.get("title") or "") + " " + (r.get("abstract") or "")).strip()
    abst = r.get("abstract") or ""
    d = {"id": r["id"]}
    for k, rx in RX.items():
        d[k] = int(bool(rx.search(text)))
    d["n_topics"] = rawmap.get(r["id"], {}).get("n_topics", 0)
    d["flesch"] = round(float(flesch(abst)), 1) if abst else 0.0
    rows.append(d)
v3 = pd.DataFrame(rows)

# ---- 주제 hotness (출판수 기반, 누수 없음) ----
sf = feat[["id", "pub_year", "topic_subfield"]].copy()
size = sf.groupby("topic_subfield")["id"].count().rename("subfield_size")
yr = sf.groupby(["topic_subfield", "pub_year"])["id"].count().rename("subfield_year_count").reset_index()
yr_prev = yr.copy(); yr_prev["pub_year"] = yr_prev["pub_year"] + 1
yr_prev = yr_prev.rename(columns={"subfield_year_count": "prev_count"})
sf = sf.merge(yr, on=["topic_subfield", "pub_year"], how="left")
sf = sf.merge(yr_prev[["topic_subfield", "pub_year", "prev_count"]], on=["topic_subfield", "pub_year"], how="left")
sf = sf.merge(size, on="topic_subfield", how="left")
sf["subfield_growth"] = sf["subfield_year_count"] / (sf["prev_count"].fillna(0) + 1)
v3 = v3.merge(sf[["id", "subfield_size", "subfield_year_count", "subfield_growth"]], on="id", how="left")

out = os.path.join(DATA, "features_v3extra.parquet")
v3.to_parquet(out, index=False)
print(f"textstat: {HAVE_TS} | saved: {out} {v3.shape}")
print(v3[["is_rct","is_sr_meta","is_cohort","is_case_report","sw_novel","sw_ai","sw_guideline"]].mean().round(3).to_dict())
print("n_topics median:", v3.n_topics.median(), "| flesch median:", v3.flesch.median(),
      "| subfield_growth median:", round(v3.subfield_growth.median(),2))
