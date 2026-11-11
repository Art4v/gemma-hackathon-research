#!/usr/bin/env bash
# VMEM pre-filter on v7 scaled items: run.sh <git-ref> <path-to-kn_items_existing.jsonl> <tag>
set -eu
SC=/private/tmp/claude-501/-Users-aarav-Documents-GitHub-gemma-hackathon-research/66df349f-ad0e-439d-9993-9c5ca7c606ac/scratchpad
D=$SC/v7bb/$3; mkdir -p $D
cd /Users/aarav/Documents/GitHub/gemma-hackathon-research && git fetch -q origin "${1#origin/}" && git show "$1:$2" > $D/src.jsonl
/usr/bin/python3 -I -c "
import json,sys
src,out=sys.argv[1:]
rows=[json.loads(l) for l in open(src) if l.strip()]
open(out,'w').write(''.join(json.dumps({'id':r['id'],'code':r['code'],'refs':{'scaled':r['ref_code']}})+'\n' for r in rows))
print(len(rows),'items')" $D/src.jsonl $D/items.jsonl
cd $SC/v3 && /usr/bin/python3 -I dataset/scripts/v3_block_bytes.py --items $D/items.jsonl --out $D/out.jsonl --python $SC/venv312/bin/python
/usr/bin/python3 -I -c "
import json,sys,collections
rows=[json.loads(l) for l in open(sys.argv[1])]; M=2**20
c=collections.Counter(); out=[]
for r in rows:
    t,l=r.get('scaled_total'),r.get('scaled_limit')
    if t is None: c['trace_error']+=1; v='trace_error'
    else:
        lim=l or 32*M
        v='over' if t>lim else ('at_risk' if t>0.9*lim else 'fits')
        c[v]+=1
    out.append({'id':r['id'],'verdict':v,'total_mib':round(t/M,2) if t else None,'limit_mib':round(l/M,2) if l else None,'blocks2x_mib':round(r['scaled']/M,2) if r.get('scaled') else None,'scratch_mib':round(r['scaled_scratch']/M,2) if r.get('scaled_scratch') is not None else None,'error':r.get('scaled_error')})
open(sys.argv[2],'w').write(''.join(json.dumps(x)+'\n' for x in out))
print(dict(c)); print('sets a limit:',sum(r.get('scaled_limit') is not None for r in rows))
" $D/out.jsonl $D/prefilter.jsonl
