#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,math,pathlib,re,time
from collections import Counter
import numpy as np
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer
from rdx_semantic_recall_stress_v2 import CORPUS, QUERIES, MODEL_ID

DENSE_WEIGHT=0.9
LEXICAL_WEIGHT=0.1
DEV_RUN_ID=36150126260
DEV_RECEIPT_SHA256="a5b40f6e97bcb2c176778e1841d03dfb639bc9c2e952ebc172b7df2f10a6bc24"
STOP={"a","an","the","our","be","to","of","in","on","and","or","is","are","can","how","what","which","why","does","from","into","its","while","must","should","with","without","real","world"}

def toks(s):
    return [x for x in re.findall(r"[a-z0-9]+",s.lower()) if x not in STOP and len(x)>1]

def bm25_scores(query,docs,k1=1.5,b=0.75):
    dts=[toks(x) for x in docs]; q=toks(query); n=len(dts)
    avdl=sum(map(len,dts))/max(n,1)
    df=Counter()
    for d in dts:
        for t in set(d): df[t]+=1
    out=[]
    for d in dts:
        tf=Counter(d); score=0.0
        for t in q:
            if t not in df: continue
            idf=math.log(1+(n-df[t]+0.5)/(df[t]+0.5))
            f=tf[t]
            score+=idf*(f*(k1+1))/(f+k1*(1-b+b*len(d)/max(avdl,1e-9)))
        out.append(score)
    return np.asarray(out,dtype=float)

def normalize(v):
    lo=float(np.min(v)); hi=float(np.max(v))
    if hi-lo<1e-12: return np.zeros_like(v,dtype=float)
    return (v-lo)/(hi-lo)

def metrics(matrix):
    id_to_idx={x["id"]:i for i,x in enumerate(CORPUS)}
    top1=top3=0; rr=0.0; rows=[]
    for qi,q in enumerate(QUERIES):
        order=np.argsort(-matrix[qi]); target=id_to_idx[q["target"]]
        rank=int(np.where(order==target)[0][0])+1
        pred=CORPUS[int(order[0])]["id"]
        top1+=int(rank==1); top3+=int(rank<=3); rr+=1.0/rank
        rows.append({"id":q["id"],"target":q["target"],"top1":pred,"target_rank":rank,
                     "top1_correct":rank==1,"top3_correct":rank<=3,
                     "query_sha256":hashlib.sha256(q["q"].encode()).hexdigest()})
    n=len(QUERIES)
    return {"correct":top1,"accuracy":top1/n,"top3_correct":top3,"top3_recall":top3/n,"mrr":rr/n,"rows":rows}

def main():
    t0=time.time()
    rev=HfApi().model_info(MODEL_ID).sha
    model=SentenceTransformer(MODEL_ID,device="cpu")
    docs=[x["text"] for x in CORPUS]; qs=[x["q"] for x in QUERIES]
    dv=np.asarray(model.encode(docs,normalize_embeddings=True,show_progress_bar=False))
    qv=np.asarray(model.encode(qs,normalize_embeddings=True,show_progress_bar=False))
    dense=qv@dv.T
    dense_norm=np.vstack([normalize(x) for x in dense])
    lexical=np.vstack([normalize(bm25_scores(q,docs)) for q in qs])
    hybrid=DENSE_WEIGHT*dense_norm+LEXICAL_WEIGHT*lexical
    d=metrics(dense); h=metrics(hybrid)
    gain=h["correct"]-d["correct"]; mrr_gain=h["mrr"]-d["mrr"]
    decision="FRESH_HOLDOUT_GAIN" if gain>0 and mrr_gain>=0 else "HOLD_NO_FRESH_GAIN"
    if gain<0 or mrr_gain<0: decision="REJECT_POLICY_FRESH_REGRESSION"
    out={
      "schema":"F115_RDX_HYBRID_RETRIEVAL_FRESH_HOLDOUT_V1",
      "status":"PASS",
      "farm_id":115,
      "model_id":MODEL_ID,
      "resolved_model_revision":rev,
      "policy_source":{"dev_run_id":DEV_RUN_ID,"dev_receipt_sha256":DEV_RECEIPT_SHA256},
      "policy":{"dense_weight":DENSE_WEIGHT,"lexical_weight":LEXICAL_WEIGHT,"frozen_before_holdout":True},
      "holdout_source":"RDX_SEMANTIC_RECALL_STRESS_V2_24_QUERY_FRESH_SET",
      "holdout_query_count":len(QUERIES),
      "baseline_dense":{k:v for k,v in d.items() if k!="rows"},
      "hybrid":{k:v for k,v in h.items() if k!="rows"},
      "top1_gain_over_dense":gain,
      "mrr_gain_over_dense":mrr_gain,
      "decision":decision,
      "rows":{"dense":d["rows"],"hybrid":h["rows"]},
      "training_executed":False,
      "weights_changed":False,
      "claim_ceiling":"FRESH_SYNTHETIC_HOLDOUT_RETRIEVAL_POLICY_EVIDENCE_ONLY_NOT_PRODUCTION_RAG_QUALITY",
      "elapsed_s":round(time.time()-t0,3)
    }
    out["receipt_sha256"]=hashlib.sha256(json.dumps(out,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path("artifacts/rdx_hybrid_retrieval_fresh_holdout_v1.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"baseline_dense":out["baseline_dense"],"hybrid":out["hybrid"],"top1_gain":gain,"mrr_gain":mrr_gain,"decision":decision,"receipt_sha256":out["receipt_sha256"]},sort_keys=True))

if __name__=="__main__":
    main()
