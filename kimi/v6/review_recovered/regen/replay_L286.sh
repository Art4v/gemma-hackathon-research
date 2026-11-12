cd /Users/hrudaynara/Projects/gemma-work/a6_review_recovered/regen/review/adv; cat > verdicts.py <<'EOF'
import json
rows=[json.loads(l) for l in open('../exclusions.jsonl')]
B={r['problem_id']:r for r in rows if r['class']=='B'}
KI={  # keep excluded: instance-level
 'kb/l1/97_ScaledDotProductAttention':'bare non-causal SDPA on given Q/K/V = 1p human kernel flash_attention(causal=False); on the team\'s own P_TASK_TWINS list (port_kernelbench.py:27-30)',
 'kernelbook/ScaleDotProductAttention_8484':'bare non-causal SDPA on given Q/K/V (S=128): 1p with causal=False',
 'kernelbook/NLgate_13812':'bare unscaled non-causal SDPA (100% of FLOPs) + q residual: 1p with causal=False, sm_scale=1',
 'authored/attention_softcap/gemma2_softcap_causal_attention':'1p signature (q,k,v)(B,H,S,D), causal, scale + one tanh softcap line; soft_cap is a flag of the 6p/7p human kernels',
 'authored/attention_softcap/gemma2_softcap_causal_attention_2':'same as gemma2_softcap_causal_attention (f32 inputs)',
 'authored/attention_softcap/gemma2_softcap_causal_attention__m1_aux':'parent is 1p+softcap; aux row statistic only',
 'authored/attention_softcap/gemma2_softcap_causal_attention_2__m1_aux':'parent is 1p+softcap; aux row statistic only',
 'authored/attention_alibi/alibi_causal_mha':'causal MHA + additive per-head (S,S) position bias = 5p reference (attn + rel_pos_bias, causal); = 1p flash_attention(ab=bias, causal=True)',
 'authored/attention_alibi/alibi_causal_mha__m1_aux':'parent is a 5p instance; aux output only',
 'authored/attention_relative_bias/t5_relative_bias_attention':'additive bucketed bias, no mask: 5p minus the causal mask; = 1p flash_attention(ab=bias, causal=False)',
 'authored/attention_relative_bias/t5_relative_bias_attention_2':'same as t5_relative_bias_attention',
 'authored/attention_sliding_window/gemma3_sliding_window_gqa_attention':'causal GQA in 2p/4p layout (B,S,H,D) with a local window: the 4p splash kernel with LocalMask instead of CausalMask (7p kernel: sliding_window flag)',
 'authored/attention_sliding_window/gemma3_sliding_window_gqa_attention__m1_aux':'parent is a 4p instance; aux output only',
 'scraped/maxtext/ragged_attention_v2':'GQA single-token decode with per-sequence lengths = 6p math with an identity page table (also returns m,l by-products)',
 'scraped/maxtext/ragged_attention_v2__m1_batch':'parent is a 6p instance',
 'authored/attention_decode_mqa/mqa_decode_padded_lengths':'decode attention with kv_lens mask = 6p with Hkv=1 and an identity page table',
 'authored/attention_decode_mqa/mqa_decode_padded_cache':'same as mqa_decode_padded_lengths',
 'authored/attention_decode_mqa/mqa_decode_padded_lengths_2':'same as mqa_decode_padded_lengths',
 'authored/attention_decode_mqa/mqa_decode_padded_cache__m1_aux':'parent is a 6p instance',
 'authored/attention_decode_mqa/mqa_decode_padded_lengths__m1_aux':'parent is a 6p instance',
 'authored/attention_chunked_prefill/chunked_prefill_attention':'offset-causal chunk attention over a cache = 7p per-sequence math (q_span=kv_len-q_len+i) with uniform q_len and identity pages',
 'authored/attention_chunked_prefill/chunked_prefill_attention__m1_aux':'parent is a 7p instance',
 'authored/attention_chunked_prefill/chunked_prefill_cache_append_attention':'KV-cache write + offset-causal attention, returns (out, k_cache, v_cache) = 60p RPAv3 (update cache + ragged attention, returns output and cache) with identity pages',
 'authored/attention_chunked_prefill/chunked_prefill_kv_offset_attention':'same as chunked_prefill_cache_append_attention (60p minus paging)',
 'authored/cross_entropy_chunked_softcap/fused_linear_softcap_cross_entropy':'64p (chunked LM-head GEMM, block 512, LSE - target logit) + one tanh softcap; vocab GEMM is ~100% of FLOPs',
 'authored/cross_entropy_chunked_softcap/fused_linear_softcap_cross_entropy_2':'same as fused_linear_softcap_cross_entropy',
 'authored/cross_entropy_chunked_softcap/fused_linear_softcap_cross_entropy_3':'same as fused_linear_softcap_cross_entropy',
 'kernelbook/SimpleCNN32Filter_12057':'conv2d+bias+ReLU is 98.7% of FLOPs: 18k chain minus its trailing bias add (the audit\'s own A standard for 41k ReLU stacks)',
 'kernelbook/convnet_3957':'conv2d(3x3)+bias+ReLU is 97% of FLOPs: 18k chain (audit\'s A standard)',
 'kernelbook/Encoder1_14018':'conv2d(3x3)+bias+ReLU is 99.5% of FLOPs: 18k chain (audit\'s A standard)',
 'kernelbook/DepthwiseSeparableConv_8314':'pointwise GEMM+bias+ReLU is 99.6% of FLOPs: exactly 41k\'s chain (audit\'s A standard), depthwise prologue 0.4%',
 'kernelbook/CAModel_14148':'1x1-conv(=GEMM)+bias+ReLU is 75% of FLOPs: 41k chain (audit\'s A standard, cf. MLP_8357 at 75%)',
 'kernelbook/MultiHeadAttention_15184':'non-causal MHA with a dense additive mask/bias, attention core 2/3 of FLOPs: 5p minus the causal mask',
 'kernelbook/AttentionUnit_13009':'non-causal single-head SDPA (S=4096,d=128) is 89% of FLOPs: 1p core with causal=False',
 'kernelbook/GlobalNonLocal_17334':'non-causal SDPA core 80% of FLOPs: 1p with causal=False',
 'kernelbook/PAM_Module_15973':'non-causal SDPA core 80% of FLOPs: 1p with causal=False',
 'kernelbook/PositionalAttentionModule_7559':'non-causal SDPA core 84% of FLOPs: 1p with causal=False',
 'kernelbook/SANet_5899':'non-causal SDPA core 80% of FLOPs: 1p with causal=False',
 'kernelbook/SelfAttention_2808':'non-causal SDPA core 87% of FLOPs: 1p with causal=False',
 'kernelbook/Attention_15176':'non-causal SDPA core 89% of FLOPs: 1p with causal=False',
 'kernelbook/PAM_Module_4494':'non-causal SDPA core 84% of FLOPs: 1p with causal=False',
 'kernelbook/SelfAttention_10797':'non-causal 8-head SDPA core 74% of FLOPs: 1p with causal=False',
}
KU={  # keep excluded: unsure (default)
 'authored/attention_softcap/gemma2_softcap_sliding_window_attention':'1p + window + softcap: one call of the 7p RPA human kernel (sliding_window, soft_cap flags)',
 'authored/attention_softcap/gemma2_softcap_sliding_window_attention__m1_aux':'parent unsure',
 'authored/attention_local_global/local_global_head_attention':'per-head causal/window masks + softcap on given Q/K/V: splash MultiHeadMask (4p kernel) config + softcap',
 'authored/attention_local_global/mixed_local_global_head_attention':'same as local_global_head_attention',
 'authored/attention_local_global/hybrid_local_global_head_attention':'same as local_global_head_attention',
 'authored/attention_local_global/local_global_head_attention__m1_aux':'parent unsure',
 'authored/attention_local_global/mixed_local_global_head_attention__m1_aux':'parent unsure',
 'authored/attention_local_global/hybrid_local_global_head_attention__m1_aux':'parent unsure',
 'authored/attention_sliding_window/gpt_oss_sliding_window_attention_sinks':'causal GQA + window (4p LocalMask) + attention sinks (52p/54p attention_sinks): two flag-level edits',
 'authored/attention_sliding_window/gpt_oss_sliding_window_sink_attention':'same as gpt_oss_sliding_window_attention_sinks',
 'authored/attention_gqa_sink/gqa_sink_sliding_window_attention':'window + sinks on given GQA Q/K/V: two flag-level edits of 2p/4p',
 'authored/attention_gqa_sink/gqa_sink_sliding_window_attention__m1_aux':'parent unsure',
 'authored/attention_gqa_sink/gqa_sink_sliding_window_specdecode':'7p-style spec-decode over a cache with window (7p flag) + sinks, no paging',
 'authored/attention_differential/differential_attention':'(A1-lam*A2)V = two causal 1p attentions sharing V (100% of FLOPs) + RMSNorm epilogue',
 'authored/attention_differential/differential_attention_2':'same as differential_attention',
 'authored/attention_differential/diff_attention_causal_subln':'same as differential_attention',
 'authored/attention_differential/differential_attention__m1_aux':'parent unsure',
 'authored/attention_differential/differential_attention_2__m1_aux':'parent unsure',
 'kernelbook/NonLocal_3178':'non-causal SDPA core exactly 2/3 of FLOPs (threshold)',
 'kernelbook/SANet_8218':'non-causal SDPA core exactly 2/3 of FLOPs (threshold)',
 'kernelbook/MultiHeadAttention_7344':'non-causal SDPA (V=K) core exactly 2/3 of FLOPs (threshold)',
 'kernelbook/Multihead_Attention_Layer_910':'non-causal SDPA core exactly 2/3 of FLOPs (threshold)',
 'kernelbook/SelfAttention_1864':'non-causal SDPA core exactly 2/3 of FLOPs (threshold)',
 'kb/l1/7_Matmul_with_small_K_dimension_':'plain GEMM (32768x64x32768); shape-only difference from 8p; on the team\'s P_TASK_TWINS list',
 'kb/l1/9_Tall_skinny_matrix_multiplication_':'plain GEMM (32768x32x32768); shape-only difference from 8p; on the team\'s P_TASK_TWINS list',
 'kernelbook/NetEnd_17238':'1x1 conv(=GEMM)+bias+softmax over output channels = 38k chain; only sizes (K=21,N=4) differ',
 'authored/norm_gated/mamba2_gated_rmsnorm_2':'12p RMSNorm (last axis) x weight + silu(gate) multiply; 12p chain is 2/3 of bytes',
 'authored/norm_gated/mamba2_gated_rmsnorm_3':'same as mamba2_gated_rmsnorm_2',
 'prod/qwen2moe_shared_expert_gate':'no reference; 10p plus a shared expert, so 10p chain likely >=2/3 of FLOPs',
 'prod/llama4_moe_input_scaled':'no reference; top-1 10p variant plus shared expert',
 'prod/jsd_distillation_loss':'no reference; LM-head GEMM(s)+softmax over vocab are 64p/13p\'s dominant stages',
 'prod/forgetting_attention':'no reference; causal attention + additive (data-dependent) bias = 1p flash_attention(ab=...)/5p form',
 'prod/parallax_attention':'no reference; described as 1p plus a second PV accumulation',
 'prod/deltanet_chunk_wy':'no reference; 63p GDN minus the decay gate (one edit)',
 'kernelbook/SelfAttention_5795':'two bias-GEMM projections (80% of FLOPs) + Q@K^T, no nonlinearity: plain-GEMM chain (8p) by the audit\'s plain-linear standard',
}
SAFE_NOTES={
 'kb/l1/95_CrossEntropyLoss':'SAFE (but reverses the team\'s P_TASK_TWINS call): plain CE over given logits; 13p/64p\'s vocab GEMM (>99% of FLOPs) is absent',
 'kernelbook/MultiheadAttention_17488':'SAFE: bare SDPA but must also write the (8,4,2048,2048) probability matrix (16x the output bytes); not a flash kernel',
 'kernelbook/Attention_2381':'SAFE: returns the (16,2048,2048) attention weights; P write dominates bytes',
 'kernelbook/GlobalAttentionGeneral_8369':'SAFE: returns the attention map; 128 keys',
 'kernelbook/GlobalAttentionGeneral_5169':'SAFE: returns the attention map; 128 keys; context projection',
 'kernelbook/CPAMDec_10410':'SAFE (borderline): attention core 65.6% (<2/3), 256 keys',
 'authored/attention_alibi/bloom_alibi_padded_attention_block_2':'SAFE: QKV/out projections are 80% of FLOPs; attention core 20%',
 'authored/attention_alibi/bloom_alibi_padded_attention_block_2__m1_aux':'SAFE: as parent',
 'authored/attention_cross/encdec_cross_attention_key_padding':'SAFE: projections 62.5% of FLOPs; key-padding cross-attention core 37.5%',
 'authored/attention_cross/padded_cross_attention':'SAFE: projections 60%',
 'authored/attention_cross/encdec_cross_attention_padmask':'SAFE: projections 74.5%',
 'authored/attention_cross/encdec_cross_attention_padmask__m1_splitk':'SAFE: projections 72%',
 'authored/attention_gqa_sink/gqa_sink_ring_cache_spec_verify':'SAFE (riskier): ring-buffer modular addressing + window + sinks + 4-token verify',
 'kernelbook/GEGLU_11635':'SAFE (riskier): GELU-GLU gate/up only (no down projection, 1/3 of 9p FLOPs) and GELU for SiLU',
 'authored/moe_expert_ffn_batched/moe_expert_ffn_batched_2':'SAFE (riskier): two per-expert batched GEMMs with GELU (dispatched tokens), not 10p\'s dense SwiGLU over all experts',
 'kernelbook/LinearEncoder_2787':'SAFE (riskier): GEMM+bias+ReLU (~100% FLOPs) with a LayerNorm epilogue; by the audit\'s FLOP standard it would be A',
 'kernelbook/LinearEmbedder_13109':'SAFE (riskier): GEMM+bias with a row-L2-normalise epilogue',
 'kernelbook/AR_12699':'SAFE: GEMV (N=1); the team\'s twin list left KernelBench L1 #4 matvec out',
 'kernelbook/AR_9990':'SAFE: GEMV (N=1)',
 'kb/l2/70_Gemm_Sigmoid_Scaling_ResidualAdd':'SAFE (riskier): 30k chain + one sigmoid; distinct KernelBench L2 problem',
 'kb/l2/27_Conv3d_HardSwish_GroupNorm_Mean':'SAFE (riskier): 25k chain + HardSwish and a spatial-only final mean; distinct KernelBench L2 problem',
 'kb/l2/73_Conv2d_BatchNorm_Scaling':'SAFE (riskier): 34k minus Mish; distinct KernelBench L2 problem',
}
out=[]
for pid,r in B.items():
    if pid in KI: v,why='keep excluded (instance)',KI[pid]
    elif pid in KU: v,why='keep excluded (unsure, default)',KU[pid]
    else: v,why='safe to re-admit',SAFE_NOTES.get(pid,'distinct: '+r['justification'][:140])
    out.append(dict(problem_id=pid, closest=r['closest_jaxbench_task'], audit_strength=r['b_strength'], in_kimi_pool_now=r['in_kimi_pool_now'], source=r['source'], verdict=v, reason=why))
missing=[k for k in list(KI)+list(KU) if k not in B]
assert not missing, missing
with open('adversarial_verdicts.jsonl','w') as f:
    for o in sorted(out,key=lambda o:(o['verdict'],o['closest'],o['problem_id'])): f.write(json.dumps(o)+'\n')
import collections
print(collections.Counter(o['verdict'] for o in out))
print(collections.Counter((o['verdict'],o['in_kimi_pool_now']) for o in out))
print(collections.Counter((o['verdict'],o['audit_strength']) for o in out))
EOF
python3 -I verdicts.py