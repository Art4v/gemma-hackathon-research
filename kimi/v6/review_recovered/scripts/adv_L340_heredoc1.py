import refs,re
for pid in ['kernelbook/PositionEmbedder_11333','kernelbook/QueryEncoding_4551','kernelbook/Join_17327','kernelbook/ItemInferenceNetwork_12467']:
    b=refs.body(refs.REF[pid]); b=re.sub(r"def create_inputs[\s\S]*?(?=\ndef |\Z)","",b); b=re.sub(r"\n\s*'name'.*","",b)
    print('#####',pid); print(b[:500])
