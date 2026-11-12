cd /private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad/review/adv; python3 - <<'EOF'
import refs,re
for pid in ['kernelbook/PositionEmbedder_11333','kernelbook/QueryEncoding_4551','kernelbook/Join_17327','kernelbook/ItemInferenceNetwork_12467']:
    b=refs.body(refs.REF[pid]); b=re.sub(r"def create_inputs[\s\S]*?(?=\ndef |\Z)","",b); b=re.sub(r"\n\s*'name'.*","",b)
    print('#####',pid); print(b[:500])
EOF
sed -n 1,72p /Users/hrudaynara/Projects/gemma-hackathon-research/control/refs/57p_SparseCore_Ragged_Gather.py | grep -v "^\s*#" | grep -v "^\s*$" | head -40