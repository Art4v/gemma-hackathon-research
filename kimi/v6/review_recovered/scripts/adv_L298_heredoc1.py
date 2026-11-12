import refs,re
for pid in ["kernelbook/SetConv_1791","kernelbook/DQN_hot1_17174","kernelbook/Attention_8837"]:
    b=refs.body(refs.REF[pid]); b=re.sub(r"def create_inputs[\s\S]*?(?=\ndef |\Z)","",b); b=re.sub(r"\n\s*'name'.*","",b)
    print('#####',pid); print(b[:1200])
