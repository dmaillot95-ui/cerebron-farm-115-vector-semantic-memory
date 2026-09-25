#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,math,pathlib,re,time
from collections import Counter

import numpy as np
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer

MODEL_ID="sentence-transformers/all-MiniLM-L6-v2"
CORPUS=[
  {"id":"RDX-S01-CLAIM-001","domain":"space","text":"Orbital assembly can reduce launch fairing constraints by joining modular structural elements after deployment."},
  {"id":"RDX-S02-CLAIM-001","domain":"space","text":"A reusable tug can move cargo between staging orbits and reduce repeated propulsion hardware per payload."},
  {"id":"RDX-AI-CLAIM-001","domain":"ai-memory","text":"Retrieval augmented generation can improve access to external knowledge without changing neural network weights."},
  {"id":"RDX-AI-CLAIM-002","domain":"ai-memory","text":"A cold benchmark must remain excluded from training data to preserve evaluation integrity."},
  {"id":"RDX-ENERGY-CLAIM-001","domain":"energy","text":"Thermal storage can shift energy availability across time without creating additional primary energy."},
  {"id":"RDX-ENERGY-CLAIM-002","domain":"energy","text":"Round trip efficiency must include both charging and discharging losses."},
  {"id":"RDX-ROBOT-CLAIM-001","domain":"robotics","text":"Magnetic or mechanical anchoring can reduce propellant use for robots operating on a large spacecraft hull."},
  {"id":"RDX-ROBOT-CLAIM-002","domain":"robotics","text":"A manipulator needs force limits and contact sensing to reduce damage during autonomous assembly."},
  {"id":"RDX-MODEL-CLAIM-001","domain":"simulation","text":"Simulation results are not physical validation and must preserve assumptions boundary conditions and uncertainty."},
  {"id":"RDX-MODEL-CLAIM-002","domain":"simulation","text":"Sensitivity analysis identifies which uncertain parameters most strongly affect model outputs."},
  {"id":"RDX-EVID-CLAIM-001","domain":"evidence","text":"A claim should be linked to provenance tests limitations and immutable evidence receipts."},
  {"id":"RDX-EVID-CLAIM-002","domain":"evidence","text":"Independent evidence must not be manufactured by counting duplicated outputs as separate proofs."}
]
QUERIES=[
  {"q":"Can external retrieval help an AI answer from a knowledge base without retraining its weights?","target":"RDX-AI-CLAIM-001"},
  {"q":"Why must our sealed evaluation set never be mixed into training?","target":"RDX-AI-CLAIM-002"},
  {"q":"How can robots stay attached to a spacecraft surface while saving maneuvering fuel?","target":"RDX-ROBOT-CLAIM-001"},
  {"q":"What analysis shows which uncertain inputs dominate a simulation result?","target":"RDX-MODEL-CLAIM-002"},
  {"q":"Does a successful simulation count as a real-world physical test?","target":"RDX-MODEL-CLAIM-001"},
  {"q":"What should connect a technical claim to trustworthy supporting material?","target":"RDX-EVID-CLAIM-001"},
  {"q":"How can building structures in orbit avoid launcher fairing size limits?","target":"RDX-S01-CLAIM-001"},
  {"q":"Which losses must be included when reporting storage round trip efficiency?","target":"RDX-ENERGY-CLAIM-002"}
]
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
    top1=0; rr=0.0; rows=[]
    for qi,q in enumerate(QUERIES):
        order=np.argsort(-matrix[qi])
        target=id_to_idx[q["target"]]
        rank=int(np.where(order==target)[0][0])+1
        pred=CORPUS[int(order[0])]["id"]
        ok=pred==q["target"]
        top1+=int(ok); rr+=1.0/rank
        rows.append({"query_sha256":hashlib.sha256(q["q"].encode()).hexdigest(),"target":q["target"],"top1":pred,"target_rank":rank,"top1_correct":ok})
    return {"correct":top1,"accuracy":top1/len(QUERIES),"mrr":rr/len(QUERIES),"rows":rows}

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

    candidates=[]
    for alpha in [1.0,0.9,0.8,0.7,0.6,0.5,0.4]:
        fused=alpha*dense_norm+(1-alpha)*lexical
        m=metrics(fused)
        candidates.append({"dense_weight":alpha,"lexical_weight":1-alpha,**m})
    best=max(candidates,key=lambda x:(x["correct"],x["mrr"],x["dense_weight"]))
    baseline=next(x for x in candidates if x["dense_weight"]==1.0)
    gain=best["correct"]-baseline["correct"]
    out={
      "schema":"F115_RDX_HYBRID_RETRIEVAL_DEV_V1",
      "status":"PASS",
      "farm_id":115,
      "model_id":MODEL_ID,
      "resolved_model_revision":rev,
      "dev_query_count":len(QUERIES),
      "baseline_dense":{"correct":baseline["correct"],"accuracy":baseline["accuracy"],"mrr":baseline["mrr"]},
      "best_policy":{"dense_weight":best["dense_weight"],"lexical_weight":best["lexical_weight"],"correct":best["correct"],"accuracy":best["accuracy"],"mrr":best["mrr"]},
      "dev_gain_over_dense_top1":gain,
      "candidates":[{k:v for k,v in x.items() if k!="rows"} for x in candidates],
      "best_rows":best["rows"],
      "decision":"CONTINUE_TO_FRESH_HOLDOUT" if gain>0 else "HOLD_NO_DEV_GAIN",
      "training_executed":False,
      "weights_changed":False,
      "claim_ceiling":"DEV_POLICY_SELECTION_ONLY_NO_PRODUCTION_RAG_GAIN_CLAIM_NO_HOLDOUT_CLAIM",
      "elapsed_s":round(time.time()-t0,3)
    }
    out["receipt_sha256"]=hashlib.sha256(json.dumps(out,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path("artifacts/rdx_hybrid_retrieval_dev.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"baseline":out["baseline_dense"],"best_policy":out["best_policy"],"decision":out["decision"],"receipt_sha256":out["receipt_sha256"]},sort_keys=True))

if __name__=="__main__":
    main()
