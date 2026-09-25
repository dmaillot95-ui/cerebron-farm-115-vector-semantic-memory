#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,pathlib,sys,time
import numpy as np
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer

MODEL_ID="sentence-transformers/all-MiniLM-L6-v2"

CORPUS=[
 {"id":"AI-RAG","domain":"ai","text":"Retrieval augmented generation supplies external context at inference time without modifying model weights."},
 {"id":"AI-M6","domain":"ai","text":"Cold evaluation prompts and answers must be excluded from training and tuning data."},
 {"id":"AI-PROV","domain":"ai","text":"Every reusable knowledge object should preserve provenance, content hash, rights, and evidence links."},
 {"id":"AI-GOLD","domain":"ai","text":"Only validated and rights-cleared GOLD records may become candidates for neural training datasets."},
 {"id":"AI-DEDUP","domain":"ai","text":"Duplicate records must be fused before counting evidence so repeated copies do not inflate confidence."},
 {"id":"AI-MEMTRAIN","domain":"ai","text":"Writing or retrieving memory is not neural learning because model parameters remain unchanged."},
 {"id":"SPACE-ASSEMBLY","domain":"space","text":"On-orbit assembly can bypass launch fairing size limits by joining structural modules after deployment."},
 {"id":"SPACE-TUG","domain":"space","text":"Reusable orbital tugs can transport payloads between staging orbits without duplicating full propulsion systems."},
 {"id":"SPACE-ANCHOR","domain":"space","text":"Mechanical or magnetic anchoring lets service robots hold position on a spacecraft while reducing propellant use."},
 {"id":"SPACE-SENSE","domain":"space","text":"Autonomous manipulators need contact sensing and force limits to avoid damaging hardware during assembly."},
 {"id":"ENERGY-RT","domain":"energy","text":"Round-trip storage efficiency must account for losses during both charging and discharging."},
 {"id":"ENERGY-THERMAL","domain":"energy","text":"Thermal storage shifts energy availability in time but does not create additional primary energy."},
 {"id":"SIM-REAL","domain":"simulation","text":"A simulation result is not physical validation and must not be presented as a real-world test."},
 {"id":"SIM-SENS","domain":"simulation","text":"Sensitivity analysis measures which uncertain parameters most influence model outputs."},
 {"id":"SIM-BC","domain":"simulation","text":"A reproducible simulation must record assumptions, boundary conditions, solver settings, and uncertainty."},
 {"id":"EVID-INDEP","domain":"evidence","text":"Outputs sharing the same lineage or copied source are correlated evidence and must not be counted as independent proof."},
 {"id":"EVID-FAIL","domain":"evidence","text":"Negative results and failures should be preserved when verified because they constrain future claims."},
 {"id":"EVID-CLAIM","domain":"evidence","text":"The strength of a claim must not exceed the strength of its supporting evidence."},
 {"id":"ROBOT-PLAN","domain":"robotics","text":"A robot action policy should enforce collision, force, and workspace constraints before execution."},
 {"id":"ROBOT-REPLAY","domain":"robotics","text":"Deterministic replay requires versioned state, configuration, seed, and action history."},
 {"id":"DATA-RIGHTS","domain":"data","text":"Private client data require explicit contractual rights and tenant isolation before any private training use."},
 {"id":"DATA-SECRET","domain":"data","text":"Secrets and credentials must never be admitted into model training datasets."},
 {"id":"DATA-SHA","domain":"data","text":"Content-addressed storage uses a cryptographic digest so the stored object can be verified against its bytes."},
 {"id":"DATA-FUSION","domain":"data","text":"Knowledge fusion should merge duplicate provenance while preserving contradictions and source identities."}
]

QUERIES=[
 {"id":"Q01","q":"Can I give the model facts from a database at answer time without retraining it?","target":"AI-RAG"},
 {"id":"Q02","q":"Why must the sealed test questions never be reused as examples for fine tuning?","target":"AI-M6"},
 {"id":"Q03","q":"Which record fields let us trace where a reusable claim came from and verify its integrity?","target":"AI-PROV"},
 {"id":"Q04","q":"When can validated research first become eligible to enter a model-training dataset?","target":"AI-GOLD"},
 {"id":"Q05","q":"How do we stop ten copies of one result from looking like ten confirmations?","target":"AI-DEDUP"},
 {"id":"Q06","q":"Does storing a fact in vector memory mean the neural network has learned it?","target":"AI-MEMTRAIN"},
 {"id":"Q07","q":"How can a structure larger than a launch fairing be constructed in space?","target":"SPACE-ASSEMBLY"},
 {"id":"Q08","q":"What reusable vehicle could move payloads from one orbital staging point to another?","target":"SPACE-TUG"},
 {"id":"Q09","q":"How can an exterior maintenance robot stay attached without continuously firing thrusters?","target":"SPACE-ANCHOR"},
 {"id":"Q10","q":"Which safeguards reduce the chance that a robotic arm damages a component while docking or assembling it?","target":"SPACE-SENSE"},
 {"id":"Q11","q":"What losses belong in the full efficiency number for an energy storage cycle?","target":"ENERGY-RT"},
 {"id":"Q12","q":"Can heat storage increase total primary energy, or does it mainly shift availability over time?","target":"ENERGY-THERMAL"},
 {"id":"Q13","q":"May a successful digital simulation be described as a physical experiment?","target":"SIM-REAL"},
 {"id":"Q14","q":"What study tells me which uncertain inputs dominate the variation in a model output?","target":"SIM-SENS"},
 {"id":"Q15","q":"What must be recorded so another team can reproduce the same simulation configuration?","target":"SIM-BC"},
 {"id":"Q16","q":"Are two answers from systems with the same underlying lineage independent scientific confirmations?","target":"EVID-INDEP"},
 {"id":"Q17","q":"Should failed experiments be deleted from the knowledge base if they were correctly measured?","target":"EVID-FAIL"},
 {"id":"Q18","q":"What rule prevents a conclusion from sounding stronger than its evidence allows?","target":"EVID-CLAIM"},
 {"id":"Q19","q":"What should an autonomous robot check before executing an action near hardware?","target":"ROBOT-PLAN"},
 {"id":"Q20","q":"What information is required to replay a simulation or agent episode deterministically?","target":"ROBOT-REPLAY"},
 {"id":"Q21","q":"What is required before a customer's private records may be used for tenant-specific training?","target":"DATA-RIGHTS"},
 {"id":"Q22","q":"Can passwords or API credentials ever be included in a training corpus?","target":"DATA-SECRET"},
 {"id":"Q23","q":"How can storage verify that an object still matches the exact bytes originally recorded?","target":"DATA-SHA"},
 {"id":"Q24","q":"When merging duplicate knowledge, what must be retained besides the canonical text?","target":"DATA-FUSION"}
]

def canon(x):
    return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()

def main():
    t0=time.time()
    sha=HfApi().model_info(MODEL_ID).sha
    model=SentenceTransformer(MODEL_ID,device="cpu")
    dv=np.asarray(model.encode([x["text"] for x in CORPUS],normalize_embeddings=True,show_progress_bar=False))
    qv=np.asarray(model.encode([x["q"] for x in QUERIES],normalize_embeddings=True,show_progress_bar=False))
    sim=qv@dv.T
    id2i={x["id"]:i for i,x in enumerate(CORPUS)}
    rows=[]; top1=0; top3=0; rr=0.0
    domain={}
    for qi,q in enumerate(QUERIES):
        order=np.argsort(-sim[qi])
        target=id2i[q["target"]]
        rank=int(np.where(order==target)[0][0])+1
        pred=CORPUS[int(order[0])]["id"]
        ok1=rank==1; ok3=rank<=3
        top1+=int(ok1); top3+=int(ok3); rr+=1.0/rank
        d=CORPUS[target]["domain"]
        z=domain.setdefault(d,{"queries":0,"top1_correct":0,"top3_correct":0})
        z["queries"]+=1; z["top1_correct"]+=int(ok1); z["top3_correct"]+=int(ok3)
        rows.append({
          "id":q["id"],
          "query_sha256":hashlib.sha256(q["q"].encode()).hexdigest(),
          "target":q["target"],"top1":pred,"target_rank":rank,
          "top1_correct":ok1,"top3_correct":ok3,
          "top1_score":float(sim[qi,int(order[0])]),
          "target_score":float(sim[qi,target])
        })
    n=len(QUERIES)
    metrics={"top1_accuracy":top1/n,"top3_recall":top3/n,"mrr":rr/n}
    thresholds={"top1_accuracy_min":0.80,"top3_recall_min":0.95,"mrr_min":0.88}
    passed=(metrics["top1_accuracy"]>=thresholds["top1_accuracy_min"] and
            metrics["top3_recall"]>=thresholds["top3_recall_min"] and
            metrics["mrr"]>=thresholds["mrr_min"])
    out={
      "schema":"F115_RDX_SEMANTIC_RECALL_STRESS_V2",
      "status":"PASS" if passed else "HOLD",
      "farm_id":115,
      "role":"VECTOR_SEMANTIC",
      "model_id":MODEL_ID,
      "resolved_model_revision":sha,
      "dataset_kind":"FRESH_SYNTHETIC_HOLDOUT_WITH_NEAR_DOMAIN_DECOYS",
      "corpus_count":len(CORPUS),"query_count":n,
      "corpus_sha256":hashlib.sha256(canon(CORPUS)).hexdigest(),
      "queries_sha256":hashlib.sha256(canon(QUERIES)).hexdigest(),
      "metrics":metrics,"thresholds":thresholds,"domain":domain,"rows":rows,
      "training_executed":False,"weights_changed":False,
      "elapsed_s":round(time.time()-t0,3),
      "claim_ceiling":"FRESH_SYNTHETIC_SEMANTIC_RETRIEVAL_STRESS_ONLY_NOT_PRODUCTION_RAG_QUALITY"
    }
    out["receipt_sha256"]=hashlib.sha256(canon(out)).hexdigest()
    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path("artifacts/rdx_semantic_recall_stress_v2.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"status":out["status"],"metrics":metrics,"thresholds":thresholds,"receipt_sha256":out["receipt_sha256"]},sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
