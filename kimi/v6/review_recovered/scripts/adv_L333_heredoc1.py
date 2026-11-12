import json, refs, re
P=[json.loads(l)['problem_id'] for l in open('../../kimi_pilot/full_problems.jsonl')]
fams=('ragged_segment_reduce','kv_cache_update','ragged_jagged_softmax','moe_expert_bincount')
for p in P:
    if any(f in p for f in fams) and '__m1' not in p:
        b=refs.body(refs.REF[p]); b=re.sub(r"def create_inputs[\s\S]*?(?=\ndef |\Z)","",b); b=re.sub(r"\n\s*'name'.*","",b)
        print('#####',p); print(b[b.find('def workload'):][:500])
