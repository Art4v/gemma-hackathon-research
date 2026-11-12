
import json
V=[json.loads(l) for l in open('adversarial_verdicts.jsonl')]
ke=[v for v in V if v['verdict']!='safe to re-admit']
print(json.dumps([f"{v['problem_id']} [{'instance' if 'instance' in v['verdict'] else 'unsure'}; {v['closest'].split('_')[0]}; {'in pool' if v['in_kimi_pool_now'] else 'not in pool'}]: {v['reason']}" for v in sorted(ke,key=lambda v:(('instance' not in v['verdict']),v['closest'],v['problem_id']))]))
print()
print(json.dumps({v['problem_id']:v['reason'] for v in V if v['verdict']=='safe to re-admit' and not v['reason'].startswith('distinct:')}))
