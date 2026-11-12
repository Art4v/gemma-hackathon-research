import json,collections
V={json.loads(l)['problem_id']:json.loads(l) for l in open('adversarial_verdicts.jsonl')}
rows=[json.loads(l) for l in open('../exclusions.jsonl')]
fam=[r['problem_id'] for r in rows if r['rule'].startswith('v4') or 'v4 rule-3' in r['rule'] or r.get('family','').startswith('v4')]
fam=set(r['problem_id'] for r in rows if 'v4' in r['rule'])
print('v4 family rows in table:',len(fam))
def tid2pid(t):
    # authored__attention_alibi__alibi_causal_mha -> authored/attention_alibi/alibi_causal_mha
    parts=t.split('__')
    if parts[0] in ('authored','scraped') and len(parts)>=3:
        base=parts[0]+'/'+parts[1]+'/'+parts[2]
        rest=parts[3:]
        if rest: base+='__'+'__'.join(rest)
        return base
    return t
for f in ['v3_train_capped.jsonl','v3_train.jsonl']:
    R=[json.loads(l) for l in open(f) if l.strip()]
    c=collections.Counter()
    for r in R:
        pid=tid2pid(r['task'])
        if pid in fam:
            v=V.get(pid,{}).get('verdict','C/val')
            c[v]+=1
    print(f, len(R), dict(c), sum(c.values()))
