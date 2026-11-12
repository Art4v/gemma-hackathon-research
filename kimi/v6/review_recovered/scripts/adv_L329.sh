cd /private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad; python3 -c "
import json
for l in open('kimi_pilot/full_problems.jsonl'):
    p=json.loads(l)['problem_id']
    if any(s in p for s in ('softcapped_logits','norm_gated','swiglu_variant','moe_shared_plus','moe_topk_routing','norm_residual','PixelNorm','L2Norm','SqueezeInit','RC_6332','ConvEncoder_8588','TinyDiscrim','GraphConvolution_8391','GumbelSoftmax','Fusion3_Minus','Fusion2_Minus','SkipModule_16307','Coboundary_8862')): print(p)
"; grep -n "build_full\|SOURCES" kimi_pilot/build_full.py | head; ls kimi_pilot | head -30