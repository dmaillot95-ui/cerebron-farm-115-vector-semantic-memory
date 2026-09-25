import hashlib, json, math, pathlib, sys, time
from collections import defaultdict

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

def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()

def cosine_matrix(a,b):
    a=a/np.clip(np.linalg.norm(a,axis=1,keepdims=True),1e-12,None)
    b=b/np.clip(np.linalg.norm(b,axis=1,keepdims=True),1e-12,None)
    return a@b.T

def main():
    t0=time.time()
    info=HfApi().model_info(MODEL_ID)
    resolved_sha=info.sha
    model=SentenceTransformer(MODEL_ID,device="cpu")
    docs=[x["text"] for x in CORPUS]
    qs=[x["q"] for x in QUERIES]
    doc_vec=np.asarray(model.encode(docs,normalize_embeddings=True,show_progress_bar=False))
    q_vec=np.asarray(model.encode(qs,normalize_embeddings=True,show_progress_bar=False))
    sim=cosine_matrix(q_vec,doc_vec)

    id_to_idx={x["id"]:i for i,x in enumerate(CORPUS)}
    rows=[]
    top1=0
    rr=0.0
    for qi,q in enumerate(QUERIES):
        order=np.argsort(-sim[qi])
        target=id_to_idx[q["target"]]
        rank=int(np.where(order==target)[0][0])+1
        pred=CORPUS[int(order[0])]["id"]
        ok=pred==q["target"]
        top1+=int(ok)
        rr+=1.0/rank
        rows.append({
          "query_sha256":hashlib.sha256(q["q"].encode()).hexdigest(),
          "target":q["target"],
          "top1":pred,
          "target_rank":rank,
          "top1_correct":ok,
          "top1_score":float(sim[qi,int(order[0])]),
          "target_score":float(sim[qi,target])
        })

    accuracy=top1/len(QUERIES)
    mrr=rr/len(QUERIES)
    status="PASS" if accuracy>=0.75 and mrr>=0.80 else "FAIL"
    receipt={
      "schema":"F115_RDX_SEMANTIC_RECALL_CANARY_V1",
      "status":status,
      "farm_id":115,
      "role":"VECTOR_SEMANTIC",
      "model_id":MODEL_ID,
      "resolved_model_revision":resolved_sha,
      "model_revision_recorded":bool(resolved_sha),
      "training_executed":False,
      "weights_changed":False,
      "corpus_count":len(CORPUS),
      "query_count":len(QUERIES),
      "corpus_sha256":hashlib.sha256(canonical(CORPUS)).hexdigest(),
      "queries_sha256":hashlib.sha256(canonical(QUERIES)).hexdigest(),
      "top1_correct":top1,
      "top1_accuracy":accuracy,
      "mrr":mrr,
      "threshold":{"top1_accuracy_min":0.75,"mrr_min":0.80},
      "rows":rows,
      "elapsed_s":round(time.time()-t0,3),
      "claim_ceiling":"SYNTHETIC_RDX_SEMANTIC_RETRIEVAL_CANARY_ONLY_NOT_PRODUCTION_RAG_QUALITY"
    }
    receipt["receipt_sha256"]=hashlib.sha256(canonical(receipt)).hexdigest()
    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path("artifacts/rdx_semantic_recall_canary.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,sort_keys=True))
    return 0 if status=="PASS" else 2

if __name__=="__main__":
    raise SystemExit(main())
