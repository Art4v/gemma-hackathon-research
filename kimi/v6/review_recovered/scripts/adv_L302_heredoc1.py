import json, re, collections
P={json.loads(l)['pid']:json.loads(l) for l in open('pool_scan4.jsonl')}
T={json.loads(l)['pid']:json.loads(l) for l in open('tasks_scan.jsonl')}
excl={json.loads(l)['problem_id']:json.loads(l) for l in open('../exclusions.jsonl')}
by=collections.defaultdict(list)
for l in open('kmatch.txt'):
    if not l.startswith('EQ '): continue
    pid=l.split()[1]
    for t in re.findall(r"\('(\d+k_[A-Za-z0-9_]+)', 'EQ'\)", l): by[t].append(pid)
for t in ['20k_Gemm_Multiply_LeakyReLU','21k_Gemm_Divide_Sum_Scaling','23k_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp','36k_Matmul_Sigmoid_Sum','37k_Matmul_Swish_Scaling','38k_Matmul_Dropout_Softmax','44k_Matmul_Divide_GELU','22k_Conv2d_InstanceNorm_Divide']:
    tk=[x for x in T if x.startswith(t[:3])][0]
    print('=====',t,'task chain:',T[tk]['top'][0][4])
    for pid in by[t]:
        r=P[pid]
        if r['nops']<=2:
            tops=[(k, s, ch) for k,s,b,rr,ch in r['top']]
            print(f"  {pid:58} nops={r['nops']} audit={excl.get(pid,{}).get('class','-')} {tops[:2]}")
