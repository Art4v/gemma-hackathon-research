import refs,re
for pid in ['kb/l1/39_L2Norm_','authored/norm_gated/gated_rmsnorm_per_head']:
    b=refs.body(refs.REF[pid]); b=re.sub(r"def create_inputs[\s\S]*?(?=\ndef |\Z)","",b)
    print('#####',pid); print(b[:500])
