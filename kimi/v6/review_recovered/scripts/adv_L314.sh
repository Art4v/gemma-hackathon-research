cd /private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad/review/adv; python3 - <<'EOF'
s=open('verdicts.py').read()
s=s.replace("KU={  # keep excluded: unsure (default)\n","""KU={  # keep excluded: unsure (default)
 'kernelbook/AttModel_10357':'four GEMM+bias+ReLU layers are exactly 2/3 of FLOPs (41k by the audit\\'s A standard, at threshold) + masked non-causal attention',
 'kernelbook/AttModel_13790':'same as AttModel_10357',
 'kernelbook/SetConv_1791':'eight GEMM+bias+ReLU layers are ~100% of FLOPs (41k by the audit\\'s A standard); masked mean pooling between them',
 'kernelbook/LinearEncoder_2787':'GEMM+bias+ReLU is ~100% of FLOPs (41k by the audit\\'s FLOP standard) with a LayerNorm epilogue',
""")
s=s.replace("KI={  # keep excluded: instance-level\n","""KI={  # keep excluded: instance-level
 'kernelbook/Attention_8837':'energy = relu(x@W+b) is ~100% of FLOPs (41k chain, audit\\'s A standard), then a GEMV scorer + softmax',
""")
s=s.replace(" 'kernelbook/LinearEncoder_2787':'SAFE (riskier): GEMM+bias+ReLU (~100% FLOPs) with a LayerNorm epilogue; by the audit\\'s FLOP standard it would be A',\n","")
s=s.replace(" 'kernelbook/AR_9990':'SAFE: GEMV (N=1)',"," 'kernelbook/AR_9990':'SAFE: GEMV (N=1)',\n 'kernelbook/DQN_hot1_17174':'SAFE (riskier): first layer 16384->15 GEMM+bias+ReLU is 99% of FLOPs but N=15 makes it a bandwidth-bound skinny GEMM (GEMV regime)',")
open('verdicts.py','w').write(s)
EOF
python3 verdicts.py; python3 -c "
import json
V=[json.loads(l) for l in open('adversarial_verdicts.jsonl')]
safe=[v['problem_id'] for v in V if v['verdict']=='safe to re-admit']
print(len(safe), sum(1 for v in V if v['verdict']=='safe to re-admit' and v['in_kimi_pool_now']), sum(1 for v in V if v['verdict']=='safe to re-admit' and not v['in_kimi_pool_now']))
import collections
print(collections.Counter(v['source'] for v in V if v['verdict']=='safe to re-admit' and not v['in_kimi_pool_now']))
print('pool removals:', [v['problem_id'] for v in V if v['verdict']!='safe to re-admit' and v['in_kimi_pool_now']])
"