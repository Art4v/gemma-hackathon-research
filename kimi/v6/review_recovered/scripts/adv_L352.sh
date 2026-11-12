cd /private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad/review/adv; python3 - <<'EOF'
import json
V=[json.loads(l) for l in open('adversarial_verdicts.jsonl')]
safe=sorted(v['problem_id'] for v in V if v['verdict']=='safe to re-admit')
open('final_safe_list.txt','w').write('\n'.join(safe)+'\n')
ke=[v for v in V if v['verdict']!='safe to re-admit']
print(len(safe), len(ke))
# compact verdict lines for keep-excluded
for v in sorted(ke,key=lambda v:(v['verdict'],v['closest'],v['problem_id'])):
    print(('I' if 'instance' in v['verdict'] else 'U'), 'P' if v['in_kimi_pool_now'] else '-', v['closest'][:4], v['problem_id'])
EOF