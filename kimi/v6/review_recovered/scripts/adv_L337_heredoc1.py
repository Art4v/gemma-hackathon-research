import json
R=[json.loads(l) for l in open('pool_scan4.jsonl')]
for r in R:
    p=r.get('prims',{})
    core={k:v for k,v in p.items() if k not in ('broadcast_in_dim','reshape','transpose','convert_element_type','squeeze','expand_dims','lt','add','select_n','concatenate','iota','clamp','ge','and','copy','slice','sub','mul')}
    if 'gather' in p and not r.get('total_GF') and len(core)<=2:
        print(r['pid'], core)
