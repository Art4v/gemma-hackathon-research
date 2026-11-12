cd /private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad/review/adv; python3 - <<'EOF'
import refs,re
for pid in ["kernelbook/SetConv_1791","kernelbook/DQN_hot1_17174","kernelbook/Attention_8837"]:
    b=refs.body(refs.REF[pid]); b=re.sub(r"def create_inputs[\s\S]*?(?=\ndef |\Z)","",b); b=re.sub(r"\n\s*'name'.*","",b)
    print('#####',pid); print(b[:1200])
EOF