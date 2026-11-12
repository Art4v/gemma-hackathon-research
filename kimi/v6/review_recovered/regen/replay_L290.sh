cd /Users/hrudaynara/Projects/gemma-work/a6_review_recovered/regen/review/adv; python3 -I -c "
import json
for l in open('adversarial_verdicts.jsonl'):
    o=json.loads(l)
    if o['verdict']=='safe to re-admit': print(o['closest'][:4], 'P' if o['in_kimi_pool_now'] else '-', o['audit_strength'][0], o['problem_id'])
" | awk '{print $1,$2,$3,$4}' | column -t | head -200