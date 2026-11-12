import json,collections
excl={json.loads(l)['problem_id']:json.loads(l) for l in open('../exclusions.jsonl')}
R=[json.loads(l) for l in open('pool_scan4.jsonl')]
EL={'add','sub','mul','div','max0','maxc','maxs','maxt','min0','minc','mins','mint','exp','tanh','logistic','log','log1p','select_n','integer_pow','sqrt','rsqrt','erf','neg','abs','sign','ge','gt','le','lt','eq','ne','and','or','not','pow','square','clamp','expm1','exp2','is_finite'}
def strict_relu(ch):
    seq=[c for c in ch if c not in ('convert_element_type','copy','copy_p')]
    el=[]
    for c in seq:
        if c in EL: el.append(c)
        else: break
    return el in (['max0'],['add','max0'],['max0','add'],['add','max0','add'])
res={'41k/19k (GEMM+bias+ReLU layers >=2/3 FLOPs)':[], '18k (conv2d+bias+ReLU layers >=2/3 FLOPs)':[]}
for r in R:
    if 'err' in r or not r.get('total_GF'): continue
    k=r.get('kinds',{})
    d=sum(s for kk,s,b,rr,ch in r['top'] if kk=='dot' and not b and strict_relu(ch))
    c=sum(s for kk,s,b,rr,ch in r['top'] if kk=='conv' and strict_relu(ch) and len(rr)==4)
    if d>=2/3-1e-9: res['41k/19k (GEMM+bias+ReLU layers >=2/3 FLOPs)'].append((r['pid'],round(d,3),excl.get(r['pid'],{}).get('class','-')))
    if c>=2/3-1e-9 and k.get('conv2d',0)>=2/3: res['18k (conv2d+bias+ReLU layers >=2/3 FLOPs)'].append((r['pid'],round(c,3),excl.get(r['pid'],{}).get('class','-')))
json.dump(res,open('pool_relu_stacks.json','w'),indent=1)
for k,v in res.items(): print(k,len(v),'| in audit table:',sum(1 for x in v if x[2]!='-'), '| share>=0.95:',sum(1 for x in v if x[1]>=0.95))
