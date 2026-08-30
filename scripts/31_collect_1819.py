# -*- coding: utf-8 -*-
"""2018-2019 척추 13저널 논문 수집 (API 키 사용) → raw_works.jsonl 에 append.
이후 02_features.py 재실행 시 dedup 되어 2018-2025 전체 코퍼스가 됨."""
import os, json, time, requests, importlib.util, sys
R = os.environ.get("CITATION_PREDICTOR_ROOT",
                     os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import oa_key
spec=importlib.util.spec_from_file_location("c",os.path.join(R,"scripts","01_collect_openalex.py"))
c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
DATA=os.path.join(R,"data"); BASE="https://api.openalex.org"; SEL=c.SELECT

def fetch(sid,name,wt):
    out=[]; cur="*"
    filt=f"primary_location.source.id:{sid},from_publication_date:2018-01-01,to_publication_date:2019-12-31,type:{wt}"
    while cur:
        p=oa_key.params({"filter":filt,"select":SEL,"per-page":200,"cursor":cur})
        for a in range(4):
            try:
                r=requests.get(f"{BASE}/works",params=p,timeout=60)
                if r.status_code==200: break
                time.sleep(2*(a+1))
            except Exception: time.sleep(2*(a+1))
        else: break
        d=r.json(); b=d.get("results",[])
        for w in b: w["_source_id"]=sid; w["_source_name"]=name
        out+=b; cur=d.get("meta",{}).get("next_cursor")
        if not b: break
        time.sleep(0.1)
    return out

srcs=c.resolve_source_ids(); srcs.setdefault("S4026894",{"name":"Spine (Phila Pa 1976)"})
n=0
with open(os.path.join(DATA,"raw_works.jsonl"),"a",encoding="utf-8") as fp:
    for sid,info in srcs.items():
        for wt in ("article","review"):
            ws=fetch(sid,info["name"],wt)
            for w in ws: fp.write(json.dumps(w,ensure_ascii=False)+"\n")
            n+=len(ws)
        print(f"  {info['name']}: total {n}")
print("appended 2018-2019:",n)
