cd /private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad/review/adv; python3 - <<'EOF'
import refs,re
for pid in ["kernelbook/GumbelSoftmax_9252","kernelbook/series_decomp_multi_17652","kernelbook/TinyDiscriminator_13761","kernelbook/GraphConvolution_8391","authored/moe_topk_routing/moe_router_softmax_topk_renorm","kernelbook/GlobalpoolFC_1550"]:
    b=refs.body(refs.REF[pid]); b=re.sub(r"def create_inputs[\s\S]*?(?=\ndef |\Z)","",b); b=re.sub(r"\n\s*'name'.*","",b)
    print('#####',pid); print(b[:800])
EOF