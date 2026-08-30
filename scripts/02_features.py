# -*- coding: utf-8 -*-
"""
raw_works.jsonl -> 피처 + 라벨 테이블.

설계 원칙 (투고 시점 예측):
  - FEATURE 는 '출판 시점에 알 수 있는' 것만 사용 (인용수/사후 지표 금지)
  - LABEL 은 사후 인용 기반: 같은 출판연도 코호트 내 상위 분위
출력: data/features.parquet, data/features.csv
"""
import json
import os
import re
import math
import pandas as pd

ROOT = os.environ.get("CITATION_PREDICTOR_ROOT",
                     os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw_works.jsonl")


def reconstruct_abstract(inv):
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def row_from_work(w):
    auth = w.get("authorships") or []
    countries, insts = set(), set()
    for a in auth:
        for c in (a.get("countries") or []):
            countries.add(c)
        for inst in (a.get("institutions") or []):
            if inst.get("id"):
                insts.add(inst["id"])
    title = (w.get("title") or w.get("display_name") or "").strip()
    abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
    topics = w.get("topics") or []
    t0 = topics[0] if topics else {}
    oa = w.get("open_access") or {}
    pl = w.get("primary_location") or {}

    # 사후 지표 (라벨용)
    cby = w.get("cited_by_count") or 0
    cby_year = {d["year"]: d["cited_by_count"] for d in (w.get("counts_by_year") or [])}
    pub_year = w.get("publication_year")
    # 출판 후 2년 누적 인용 (early citation; 후속 분석용 참고지표)
    c2 = sum(v for y, v in cby_year.items()
             if pub_year and y is not None and y <= pub_year + 2)
    cnp = (w.get("citation_normalized_percentile") or {}).get("value")

    return {
        "id": w.get("id", "").split("/")[-1],
        "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
        "journal": w.get("_source_name"),
        "pub_year": pub_year,
        "pub_date": w.get("publication_date"),
        "type": w.get("type"),
        "language": w.get("language"),
        "is_retracted": w.get("is_retracted"),
        # ----- FEATURES (출판 시점 가용) -----
        "n_authors": len(auth),
        "n_institutions": len(insts),
        "n_countries": len(countries),
        "is_international": int(len(countries) > 1),
        "has_korea": int("KR" in countries),
        "has_usa": int("US" in countries),
        "title_n_words": len(title.split()),
        "title_n_chars": len(title),
        "title_has_colon": int(":" in title),
        "title_has_question": int("?" in title),
        "abstract_n_words": len(abstract.split()),
        "has_abstract": int(bool(abstract)),
        "n_references": w.get("referenced_works_count") or 0,
        "is_oa": int(bool(oa.get("is_oa"))),
        "oa_status": oa.get("oa_status"),
        "is_review": int((w.get("type") == "review")),
        "topic_field": (t0.get("field") or {}).get("display_name"),
        "topic_subfield": (t0.get("subfield") or {}).get("display_name"),
        "topic_name": t0.get("display_name"),
        "pub_month": int(w["publication_date"][5:7]) if w.get("publication_date") else None,
        "title": title,
        "abstract": abstract,
        # ----- LABEL 재료 (사후) -----
        "cited_by_count": cby,
        "c2_early": c2,
        "oa_norm_percentile": cnp,
        "fwci": w.get("fwci"),
    }


def main():
    rows = []
    with open(RAW, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(row_from_work(json.loads(line)))
            except Exception as e:
                print("skip row:", e)
    df = pd.DataFrame(rows)
    print(f"raw rows: {len(df)}")

    # 중복 제거 (id 기준)
    df = df.drop_duplicates(subset="id").reset_index(drop=True)
    # article 만, 철회/언어 비영어 제외 옵션
    df = df[df["is_retracted"] != True].copy()
    df = df[df["pub_year"].notna()].copy()
    df["pub_year"] = df["pub_year"].astype(int)
    # 코퍼스 = 2018-2023 (성숙 라벨만; 2024-2025 제외 — 인용 미성숙)
    df = df[df["pub_year"].between(2018, 2023)].copy()
    print(f"after clean + year filter(2018-2023): {len(df)}")

    # 학회 초록/사설 제거: 정상 research article 은 참고문헌을 가짐.
    # (OpenAlex 가 학회 초록을 type=article 로 잘못 분류 -> n_references<5 로 걸러냄)
    before = len(df)
    df = df[df["n_references"] >= 5].copy()
    print(f"after abstract/editorial filter (n_references>=5): {len(df)} "
          f"(removed {before - len(df)})")

    # ----- 라벨 A: 분야 전체 같은 출판연도 코호트 내 백분위 (특성 분석용) -----
    df["cohort_pct"] = df.groupby("pub_year")["cited_by_count"].rank(pct=True)
    df["is_top10"] = (df["cohort_pct"] >= 0.90).astype(int)
    df["is_top25"] = (df["cohort_pct"] >= 0.75).astype(int)
    # ----- 라벨 B: 저널 내 같은 출판연도 백분위 (ASJ 투고 스코어링용) -----
    df["cohort_pct_jrnl"] = df.groupby(["journal", "pub_year"])["cited_by_count"].rank(pct=True)
    df["is_top10_jrnl"] = (df["cohort_pct_jrnl"] >= 0.90).astype(int)
    df["is_top25_jrnl"] = (df["cohort_pct_jrnl"] >= 0.75).astype(int)
    # log 인용 (회귀 보조 라벨)
    df["log_citations"] = df["cited_by_count"].apply(lambda x: math.log1p(x))
    # 연 평균 인용 (나이 보정)
    age = (2026 - df["pub_year"]).clip(lower=1)
    df["citations_per_year"] = df["cited_by_count"] / age

    out_parq = os.path.join(DATA, "features.parquet")
    out_csv = os.path.join(DATA, "features.csv")
    try:
        df.to_parquet(out_parq, index=False)
        print("saved:", out_parq)
    except Exception as e:
        print("parquet skip (install pyarrow):", e)
    df.drop(columns=["abstract"]).to_csv(out_csv, index=False)
    print("saved:", out_csv)

    # 요약
    print("\n연도별:")
    print(df.groupby("pub_year").agg(
        n=("id", "size"),
        median_cite=("cited_by_count", "median"),
        mean_cite=("cited_by_count", "mean"),
    ))
    print("\n저널별 (n):")
    print(df["journal"].value_counts())


if __name__ == "__main__":
    main()
