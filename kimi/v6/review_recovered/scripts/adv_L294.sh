cd /private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad/review/adv; python3 - <<'EOF'
import json, importlib.util, os
os.environ['JAX_PLATFORMS']='cpu'
spec=importlib.util.spec_from_file_location('sp','scan_pool4.py'); sp=importlib.util.module_from_spec(spec); spec.loader.exec_module(sp)
import refs
EL={'add','sub','mul','div','max0','maxc','maxs','maxt','min0','minc','mins','mint','exp','tanh','logistic','log','log1p','select_n','integer_pow','sqrt','rsqrt','erf','neg','abs','sign','ge','gt','le','lt','eq','ne','and','or','not','pow','square','clamp','expm1','exp2','is_finite'}
def strict_relu(ch):
    seq=[c for c in ch if c not in ('convert_element_type','copy','copy_p')]
    el=[]
    for c in seq:
        if c in EL: el.append(c)
        else: break
    return el in (['max0'],['add','max0'],['max0','add'],['add','max0','add'])
V=[json.loads(l) for l in open('adversarial_verdicts.jsonl')]
for o in V:
    c=refs.REF.get(o['problem_id'])
    if not c: continue
    try: r=sp.analyze(c)
    except Exception as e: continue
    for kind in ('dot','conv'):
        share=sum(s for k,s,b,rr,ch in r['top'] if k==kind and not b and strict_relu(ch))
        if share>=0.6:
            print(f"{o['verdict'][:22]:24} {o['problem_id']:60} {kind} relu-share={round(share,3)}")
EOF