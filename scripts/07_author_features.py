# -*- coding: utf-8 -*-
"""
저자 명성 피처 (누수 방지: 출판 '전년도'까지의 누적 실적만).

각 논문의 제1저자/교신(마지막)저자에 대해:
  - prior_works   : 출판 전년도까지 누적 발표수
  - prior_cites   : 출판 전년도까지 누적 피인용수
  - h_index_now   : 현재 h-index (참고용, 약한 누수 있음 -> 별도 표기)
OpenAlex authors API 의 counts_by_year 로 as-of-year 재구성.
출력: data/author_features.parquet (key=work id)
"""
import os
import json
import time
import requests
import sys
import pandas as pd

ROOT = os.environ.get("CITATION_PREDICTOR_ROOT",
                     os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oa_key
MAILTO = oa_key.MAILTO
BASE = "https://api.openalex.org"


def collect_author_refs():
    """work id -> (first_author_id, last_author_id, pub_year) + 전체 unique author ids."""
    refs = {}
    uniq = set()
    seen = set()
    with open(os.path.join(DATA, "raw_works.jsonl"), encoding="utf-8") as fp:
        for line in fp:
            w = json.loads(line)
            wid = (w.get("id") or "").split("/")[-1]
            if not wid or wid in seen:
                continue
            seen.add(wid)
            auth = w.get("authorships") or []
            if not auth:
                continue
            first = last = None
            for a in auth:
                aid = (a.get("author") or {}).get("id")
                if not aid:
                    continue
                aid = aid.split("/")[-1]
                pos = a.get("author_position")
                if pos == "first" and first is None:
                    first = aid
                if pos == "last":
                    last = aid
            # fallback
            if first is None:
                first = ((auth[0].get("author") or {}).get("id") or "").split("/")[-1] or None
            if last is None:
                last = ((auth[-1].get("author") or {}).get("id") or "").split("/")[-1] or None
            refs[wid] = (first, last, w.get("publication_year"))
            if first:
                uniq.add(first)
            if last:
                uniq.add(last)
    return refs, uniq


def fetch_authors(author_ids):
    """author id -> {counts_by_year, h_index}. 50개씩 OR 필터로 배치."""
    info = {}
    ids = [a for a in author_ids if a]
    B = 50
    for i in range(0, len(ids), B):
        chunk = ids[i:i + B]
        filt = "ids.openalex:" + "|".join(chunk)
        params = oa_key.params({"filter": filt, "select": "id,summary_stats,counts_by_year",
                                "per-page": B})
        for attempt in range(4):
            try:
                r = requests.get(f"{BASE}/authors", params=params, timeout=60)
                if r.status_code == 200:
                    break
                time.sleep(2 * (attempt + 1))
            except Exception:
                time.sleep(2 * (attempt + 1))
        else:
            print(f"  batch {i} failed, skip")
            continue
        for a in r.json().get("results", []):
            aid = a["id"].split("/")[-1]
            cby = {d["year"]: (d.get("works_count", 0), d.get("cited_by_count", 0))
                   for d in (a.get("counts_by_year") or [])}
            info[aid] = {
                "counts_by_year": cby,
                "h_index": (a.get("summary_stats") or {}).get("h_index", 0),
            }
        if (i // B) % 10 == 0:
            print(f"  authors {i+len(chunk)}/{len(ids)}")
        time.sleep(0.12)
    return info


def prior(info, aid, year):
    """as-of (year-1) 누적 works, cites."""
    if not aid or aid not in info or not year:
        return 0, 0
    cby = info[aid]["counts_by_year"]
    pw = sum(w for y, (w, c) in cby.items() if y <= year - 1)
    pc = sum(c for y, (w, c) in cby.items() if y <= year - 1)
    return pw, pc


def main():
    print("1) author refs 추출...")
    refs, uniq = collect_author_refs()
    print(f"   works {len(refs)}, unique authors {len(uniq)}")

    print("2) authors API 조회 (배치)...")
    info = fetch_authors(list(uniq))
    print(f"   fetched {len(info)} authors")

    print("3) as-of-year 피처 생성...")
    rows = []
    for wid, (fa, la, yr) in refs.items():
        fw, fc = prior(info, fa, yr)
        lw, lc = prior(info, la, yr)
        rows.append({
            "id": wid,
            "fa_prior_works": fw, "fa_prior_cites": fc,
            "la_prior_works": lw, "la_prior_cites": lc,
            "team_prior_works_max": max(fw, lw),
            "team_prior_cites_max": max(fc, lc),
            "la_h_index_now": info.get(la, {}).get("h_index", 0) if la else 0,
        })
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(DATA, "author_features.parquet"), index=False)
    print("saved: data/author_features.parquet", df.shape)
    print(df[["fa_prior_works", "fa_prior_cites", "la_prior_cites", "la_h_index_now"]].describe().round(1))


if __name__ == "__main__":
    main()
