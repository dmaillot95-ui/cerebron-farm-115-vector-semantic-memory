import hashlib,json,pathlib,sys
ROLE="VECTOR_SEMANTIC"; FARM=115
payload={"namespace":"MEMORY_OMEGA","class":"M1","concept_id":"MEM-DEMO-001","glyph":"MΩ:001","content":"canonical-memory-canary","producer":"F115"}
raw=json.dumps(payload,sort_keys=True,separators=(",",":")).encode();sha=hashlib.sha256(raw).hexdigest()
record={"farm":FARM,"role":ROLE,"memory_id":f"MEMORY_OMEGA:M1:{sha[:16]}","sha256":sha,"bytes":len(raw),"status":"PASS","epistemic":"MEMORY_PROTOCOL_CANARY_ONLY"}
checks={"sha64":len(sha)==64,"addressed":record["memory_id"].endswith(sha[:16]),"nonempty":record["bytes"]>0};record["checks"]=checks
record["status"]="PASS" if all(checks.values()) else "FAIL"
pathlib.Path("artifacts").mkdir(exist_ok=True);pathlib.Path("artifacts/node.json").write_text(json.dumps(record,indent=2)+"\\n");print(json.dumps(record));sys.exit(0 if record["status"]=="PASS" else 1)
