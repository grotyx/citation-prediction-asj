# -*- coding: utf-8 -*-
"""
Spine 분야 다저널 논문을 OpenAlex에서 수집.
- 대상: 척추 전문 저널 (ISSN으로 source 해석)
- 기간: 2020-2025 (최근 5년)
- 출력: data/raw_works.parquet (+ 진행상황 jsonl 캐시)

투고 시점 예측이 목표이므로 '출판 시점에 알 수 있는' 변수 위주로 가공하되,
원본 필드는 최대한 보존해 후속 단계에서 자유롭게 파생.
"""
import json
import os
import time
import sys
import requests

ROOT = os.environ.get("CITATION_PREDICTOR_ROOT",
                     os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

MAILTO = os.environ.get("OPENALEX_MAILTO", "")   # OpenAlex polite pool
BASE = "https://api.openalex.org"

# 척추 전문 저널 (ISSN-L 기준). 종합 신경외과/정형외과지는 노이즈가 커서 제외.
SPINE_JOURNALS = {
    "Spine (Phila Pa 1976)":            "0362-2436",
    "European Spine Journal":           "0940-6719",
    "The Spine Journal":                "1529-9430",
    "Global Spine Journal":             "2192-5682",
    "Journal of Neurosurgery: Spine":   "1547-5654",
    "Asian Spine Journal":              "1976-1902",
    "Clinical Spine Surgery":           "2380-0186",
    "Spine Deformity":                  "2212-134X",
    "Neurospine":                       "2586-6583",
    "Journal of Spinal Cord Medicine":  "1079-0268",
    "Spinal Cord":                      "1362-4393",
    "Brain and Spine":                  "2772-5294",
    "North American Spine Society J":   "2666-5484",
    "European Spine Journal Suppl":     "1432-0932",
}

YEAR_FROM, YEAR_TO = 2020, 2025

# 본문에서 받을 필드만 select 해서 페이로드 축소
SELECT = ",".join([
    "id", "doi", "title", "display_name", "publication_year", "publication_date",
    "type", "language", "cited_by_count", "counts_by_year",
    "referenced_works_count", "is_retracted", "authorships",
    "open_access", "primary_location", "topics", "abstract_inverted_index",
    "fwci", "citation_normalized_percentile",
])


def resolve_source_ids():
    """ISSN -> OpenAlex source id 매핑."""
    out = {}
    for name, issn in SPINE_JOURNALS.items():
        url = f"{BASE}/sources/issn:{issn}"
        try:
            r = requests.get(url, params={"mailto": MAILTO}, timeout=30)
            if r.status_code == 200:
                d = r.json()
                out[d["id"].split("/")[-1]] = {
                    "name": name,
                    "openalex_name": d.get("display_name"),
                    "works_count": d.get("works_count"),
                }
                print(f"  OK  {name:32s} -> {d['id'].split('/')[-1]} "
                      f"({d.get('display_name')}, n={d.get('works_count')})")
            else:
                print(f"  MISS {name:32s} issn={issn} status={r.status_code}")
        except Exception as e:
            print(f"  ERR {name:32s} {e}")
        time.sleep(0.2)
    return out


def fetch_source_works(source_id, source_name, work_type="article"):
    """한 저널의 2020-2025 works 를 cursor 페이지네이션으로 전부 수집."""
    works = []
    cursor = "*"
    filt = (f"primary_location.source.id:{source_id},"
            f"from_publication_date:{YEAR_FROM}-01-01,"
            f"to_publication_date:{YEAR_TO}-12-31,"
            f"type:{work_type}")
    while cursor:
        params = {
            "filter": filt,
            "select": SELECT,
            "per-page": 200,
            "cursor": cursor,
            "mailto": MAILTO,
        }
        for attempt in range(4):
            try:
                r = requests.get(f"{BASE}/works", params=params, timeout=60)
                if r.status_code == 200:
                    break
                print(f"    [{source_name}] status {r.status_code}, retry {attempt+1}")
                time.sleep(2 * (attempt + 1))
            except Exception as e:
                print(f"    [{source_name}] err {e}, retry {attempt+1}")
                time.sleep(2 * (attempt + 1))
        else:
            print(f"    [{source_name}] FAILED batch, stopping this source")
            break
        d = r.json()
        batch = d.get("results", [])
        for w in batch:
            w["_source_id"] = source_id
            w["_source_name"] = source_name
        works.extend(batch)
        cursor = d.get("meta", {}).get("next_cursor")
        print(f"    [{source_name}] +{len(batch)} (total {len(works)})")
        if not batch:
            break
        time.sleep(0.15)
    return works


def main():
    print("=== 1) source id 해석 ===")
    sources = resolve_source_ids()
    if not sources:
        print("source 해석 실패. 종료.")
        sys.exit(1)

    print(f"\n=== 2) works 수집 ({YEAR_FROM}-{YEAR_TO}) ===")
    all_works = []
    raw_path = os.path.join(DATA, "raw_works.jsonl")
    with open(raw_path, "w", encoding="utf-8") as fp:
        for sid, info in sources.items():
            print(f"\n[{info['name']}]")
            ws = fetch_source_works(sid, info["name"])
            for w in ws:
                fp.write(json.dumps(w, ensure_ascii=False) + "\n")
            all_works.extend(ws)

    print(f"\n총 수집 works: {len(all_works)}")
    print(f"원본 저장: {raw_path}")

    # 저널별 카운트 요약
    from collections import Counter
    c = Counter(w["_source_name"] for w in all_works)
    print("\n저널별:")
    for name, n in c.most_common():
        print(f"  {name:34s} {n}")


if __name__ == "__main__":
    main()
