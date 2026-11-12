
import json
V=[json.loads(l) for l in open('adversarial_verdicts.jsonl')]
safe=[v['problem_id'] for v in V if v['verdict']=='safe to re-admit']
print(len(safe), sum(1 for v in V if v['verdict']=='safe to re-admit' and v['in_kimi_pool_now']), sum(1 for v in V if v['verdict']=='safe to re-admit' and not v['in_kimi_pool_now']))
import collections
print(collections.Counter(v['source'] for v in V if v['verdict']=='safe to re-admit' and not v['in_kimi_pool_now']))
print('pool removals:', [v['problem_id'] for v in V if v['verdict']!='safe to re-admit' and v['in_kimi_pool_now']])
