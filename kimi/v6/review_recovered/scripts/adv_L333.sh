cd /Users/hrudaynara/Projects/gemma-hackathon-research/control/refs; sed -n '/^def workload/,$p' 57p_SparseCore_Ragged_Gather.py | head -30; cd /private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad/review/adv; python3 - <<'EOF'
import json, refs, re
P=[json.loads(l)['problem_id'] for l in open('../../kimi_pilot/full_problems.jsonl')]
fams=('ragged_segment_reduce','kv_cache_update','ragged_jagged_softmax','moe_expert_bincount')
for p in P:
    if any(f in p for f in fams) and '__m1' not in p:
        b=refs.body(refs.REF[p]); b=re.sub(r"def create_inputs[\s\S]*?(?=\ndef |\Z)","",b); b=re.sub(r"\n\s*'name'.*","",b)
        print('#####',p); print(b[b.find('def workload'):][:500])
EOF