cd /private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad/review/adv; python3 -c "
import json,collections
V=[json.loads(l) for l in open('adversarial_verdicts.jsonl')]
c=collections.Counter()
for v in V:
    t=v['closest'].split('_')[0]
    grp='attn(1p-7p,15p,16p,65p)' if t in ('1p','2p','4p','5p','6p','7p','15p','16p','65p') else ('CE(13p,64p)' if t in ('13p','64p') else ('k' if t.endswith('k') else 'other p'))
    c[(grp, v['verdict'][:4])]+=1
for k in sorted(c): print(k,c[k])
"; ls -la /private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad/review/adv/ | grep -v "^d" | awk '{print $5, $9}'