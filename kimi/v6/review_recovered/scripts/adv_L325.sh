cd /private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad/review/adv; python3 - <<'EOF'
import json,collections
V={json.loads(l)['problem_id']:json.loads(l) for l in open('adversarial_verdicts.jsonl')}
rows=[json.loads(l) for l in open('../exclusions.jsonl')]
fam=set(r['problem_id'] for r in rows if 'v4' in r['rule'])
def tid2pid(t):
    parts=t.split('__')
    if parts[0] in ('authored','scraped') and len(parts)>=3:
        base=parts[0]+'/'+parts[1]+'/'+parts[2]
        if parts[3:]: base+='__'+'__'.join(parts[3:])
        return base
    return t
R=[json.loads(l) for l in open('v3_train_capped.jsonl') if l.strip()]
c=collections.Counter()
for r in R:
    pid=tid2pid(r['task'])
    if pid in fam: c[(V.get(pid,{}).get('verdict','C'),pid)]+=1
for (v,p),n in sorted(c.items()):
    if v.startswith('safe'): print(n,p)
EOF