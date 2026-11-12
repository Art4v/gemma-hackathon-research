S=/private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad; cat >> $S/review/labels.py <<'EOF'
# ---------------- prod candidates dropped before authoring (prod_excluded_by_algorithm.jsonl; judged from stem + reason only) ----------------
P = "prod/"
for st_, cls, stn, task, fam, why in [
 ("deltanet_chunk_wy","B","near","63p_GDN","linear_attn_ssm","ungated delta rule (chunked WY); 63p's data-dependent gate is a whole input stage"),
 ("dplr_generalized_delta_chunk","B","far","63p_GDN","linear_attn_ssm","DPLR generalised delta rule: different state update"),
 ("deltaformer_attention","B","far","63p_GDN","attention","delta rule inside softmax attention"),
 ("ttt_linear","B","far","63p_GDN","linear_attn_ssm","TTT-linear inner-loop update"),
 ("gla_recurrent","B","far","15p_RetNet_Retention","linear_attn_ssm","gated linear attention with per-channel data-dependent decay (15p: per-head constant decay)"),
 ("log_linear_attention","B","far","16p_Mamba2_SSD","linear_attn_ssm","log-linear (hierarchical Fenwick) attention"),
 ("gated_slot_attention","B","far","15p_RetNet_Retention","linear_attn_ssm","two gated linear-attention recurrences over slots"),
 ("forgetting_attention","B","near","5p_Flex_Attention","attention","causal softmax attention + data-dependent cumulative forget-gate bias computed in-kernel (5p loads a dense bias)"),
 ("parallax_attention","B","far","1p_Flash_Attention","attention","second PV accumulation"),
 ("merge_attention_states_lse","B","far","1p_Flash_Attention","attention","only the LSE merge of two partial attention states (elementwise)"),
 ("attnres_depth_mixing","B","far","56p_DeepSeek_V4_MHC","mhc_sinkhorn","multi-stream residual depth mixing, no Sinkhorn"),
 ("gemma3n_altup","B","far","56p_DeepSeek_V4_MHC","mhc_sinkhorn","AltUp predict/correct stream mixing"),
 ("dpo_loss_fused_lm_head","A","","64p_Linear_Softmax_Cross_Entropy","cross_entropy","LM-head GEMM + log-softmax + target gather dominate; the DPO sigmoid head is trivial"),
 ("jsd_distillation_loss","B","near","64p_Linear_Softmax_Cross_Entropy","cross_entropy","full-distribution JSD between two softmaxes; no target gather"),
 ("grpo_clipped_policy_loss","A","","64p_Linear_Softmax_Cross_Entropy","cross_entropy","LM-head GEMM + log-softmax + gather dominate; clipped ratio is trivial"),
 ("orpo_loss","A","","64p_Linear_Softmax_Cross_Entropy","cross_entropy","LM-head cross-entropy dominates; odds-ratio term is trivial"),
 ("gptoss_moe_clamped_swiglu","A","","10p_Sparse_MoE","moe","top-k routed SwiGLU experts; clamp limit and alpha are constants (borderline; reference never written)"),
 ("qwen2moe_shared_expert_gate","B","near","10p_Sparse_MoE","moe","10p plus a sigmoid-gated shared expert"),
 ("llama4_moe_input_scaled","B","near","10p_Sparse_MoE","moe","top-1 routing that scales the input by the router score + shared expert"),
 ("deepseekv3_group_limited_router","B","far","10p_Sparse_MoE","moe","router only (group-limited top-k); 10p's cost is the expert GEMMs"),
 ("fused_add_rmsnorm_fp8_group_quant","B","near","12p_RMSNorm","norm","residual add + RMSNorm + fp8 per-group quantisation epilogue"),
 ("clip_contrastive_loss","B","near","64p_Linear_Softmax_Cross_Entropy","cross_entropy","symmetric CE: row-wise AND column-wise logsumexp over the same logits"),
 ("ntxent_memory_bank_loss","A","","64p_Linear_Softmax_Cross_Entropy","cross_entropy","q@memory^T + positive logit, CE with label 0 (borderline)"),
 ("infonce_explicit_negatives","B","near","64p_Linear_Softmax_Cross_Entropy","cross_entropy","per-sample explicit negatives: batched dots, not a shared weight matrix"),
 ("kivi_2bit_kv_decode","B","far","65p_Quantized_Splash_Attention","attention","2-bit asymmetric KV dequant + non-paged single-token decode"),
]: a(P+st_, cls, stn, task, fam, why)
# ---------------- v4 rule-3 families (dataset/v4/exclude_families.txt) ----------------
V = {
 "authored/attention_sliding_window/gemma3_sliding_window_gqa_attention": ("B","near","4p_Sparse_Attention","causal GQA with a sliding-window band mask"),
 "authored/attention_sliding_window/gpt_oss_sliding_window_attention_sinks": ("B","far","4p_Sparse_Attention","sliding-window GQA + learned sink logits"),
 "authored/attention_sliding_window/gpt_oss_sliding_window_sink_attention": ("B","far","4p_Sparse_Attention","sliding-window GQA + learned sink logits"),
 "authored/attention_alibi/alibi_causal_mha": ("B","near","5p_Flex_Attention","causal MHA + ALiBi bias generated in-kernel (5p streams a dense (H,S,S) bias)"),
 "authored/attention_alibi/bloom_alibi_padded_attention_block_2": ("B","far","5p_Flex_Attention","QKV/out projections + padding-aware ALiBi + causal"),
 "authored/attention_gqa_sink/gqa_sink_sliding_window_attention": ("B","far","4p_Sparse_Attention","window + sinks"),
 "authored/attention_gqa_sink/gqa_sink_ring_cache_spec_verify": ("B","far","6p_Paged_Attention","speculative verify over a ring-buffer cache with sinks"),
 "authored/attention_gqa_sink/gqa_sink_sliding_window_specdecode": ("B","far","7p_Ragged_Paged_Attention","spec-decode over ragged cache with window + sinks"),
 "scraped/maxtext/ragged_attention_v2": ("B","near","6p_Paged_Attention","GQA single-token decode with per-entry lengths over a contiguous cache (no page table)"),
 "authored/cross_entropy_chunked_softcap/fused_linear_softcap_cross_entropy": ("B","near","64p_Linear_Softmax_Cross_Entropy","tanh logit softcap between LM head and CE; per-token losses"),
 "authored/cross_entropy_chunked_softcap/fused_linear_softcap_cross_entropy_2": ("B","near","64p_Linear_Softmax_Cross_Entropy","tanh logit softcap between LM head and CE"),
 "authored/cross_entropy_chunked_softcap/fused_linear_softcap_cross_entropy_3": ("B","near","64p_Linear_Softmax_Cross_Entropy","tanh logit softcap between LM head and CE"),
 "authored/attention_softcap/gemma2_softcap_causal_attention": ("B","near","1p_Flash_Attention","causal MHA + tanh logit softcap"),
 "authored/attention_softcap/gemma2_softcap_causal_attention_2": ("B","near","1p_Flash_Attention","causal MHA + tanh logit softcap"),
 "authored/attention_softcap/gemma2_softcap_sliding_window_attention": ("B","far","1p_Flash_Attention","softcap + sliding window"),
 "authored/attention_relative_bias/t5_relative_bias_attention": ("B","far","5p_Flex_Attention","bidirectional + bucketed bias gathered from a table"),
 "authored/attention_relative_bias/t5_relative_bias_attention_2": ("B","far","5p_Flex_Attention","bidirectional + bucketed bias gathered from a table"),
 "authored/attention_cross/encdec_cross_attention_key_padding": ("B","far","1p_Flash_Attention","projections + key-padding cross-attention"),
 "authored/attention_cross/padded_cross_attention": ("B","far","1p_Flash_Attention","projections + key-padding cross-attention"),
 "authored/attention_cross/encdec_cross_attention_padmask": ("B","far","1p_Flash_Attention","projections + key-padding cross-attention"),
 "authored/attention_chunked_prefill/chunked_prefill_attention": ("B","near","7p_Ragged_Paged_Attention","offset-causal chunk over a prefix, no paging"),
 "authored/attention_chunked_prefill/chunked_prefill_cache_append_attention": ("B","far","7p_Ragged_Paged_Attention","KV-cache write + offset-causal attention"),
 "authored/attention_chunked_prefill/chunked_prefill_kv_offset_attention": ("B","far","7p_Ragged_Paged_Attention","KV-cache write + offset-causal attention"),
 "authored/attention_decode_mqa/mqa_decode_padded_lengths": ("B","near","6p_Paged_Attention","MQA decode over a contiguous padded cache (no paging)"),
 "authored/attention_decode_mqa/mqa_decode_padded_cache": ("B","near","6p_Paged_Attention","MQA decode over a contiguous padded cache"),
 "authored/attention_decode_mqa/mqa_decode_padded_lengths_2": ("B","near","6p_Paged_Attention","MQA decode over a contiguous padded cache"),
 "authored/attention_linear_elu/linear_attention_elu_noncausal": ("B","far","15p_RetNet_Retention","non-causal elu+1 linear attention"),
 "authored/attention_linear_elu/linear_attention_elu": ("B","far","15p_RetNet_Retention","non-causal elu+1 linear attention"),
 "authored/attention_linear_elu/linear_attention_elu_noncausal_2": ("B","far","15p_RetNet_Retention","non-causal elu+1 linear attention"),
 "authored/attention_differential/differential_attention": ("B","far","1p_Flash_Attention","two softmax maps (A1 - lambda*A2)V + per-head RMSNorm"),
 "authored/attention_differential/differential_attention_2": ("B","far","1p_Flash_Attention","two softmax maps + per-head RMSNorm"),
 "authored/attention_differential/diff_attention_causal_subln": ("B","far","1p_Flash_Attention","two softmax maps + SubLN (also a within-set duplicate)"),
 "authored/attention_local_global/mixed_local_global_head_attention": ("B","far","1p_Flash_Attention","per-head window/causal masks + softcap"),
 "authored/attention_local_global/local_global_head_attention": ("B","far","1p_Flash_Attention","per-head window/causal masks + softcap"),
 "authored/attention_local_global/hybrid_local_global_head_attention": ("B","far","1p_Flash_Attention","per-head window/causal masks + softcap"),
}
for pid, (cls, stn, task, why) in V.items():
    a(pid, cls, stn, task, "attention" if "cross_entropy" not in pid else "cross_entropy", why)
for parent in list(V):
    for suf in ["__m1_aux", "__m1_splitk", "__m1_batch"]:
        pass
CHILD = {"authored/attention_alibi/alibi_causal_mha__m1_aux":"authored/attention_alibi/alibi_causal_mha",
 "authored/attention_alibi/bloom_alibi_padded_attention_block_2__m1_aux":"authored/attention_alibi/bloom_alibi_padded_attention_block_2",
 "authored/attention_chunked_prefill/chunked_prefill_attention__m1_aux":"authored/attention_chunked_prefill/chunked_prefill_attention",
 "authored/attention_cross/encdec_cross_attention_padmask__m1_splitk":"authored/attention_cross/encdec_cross_attention_padmask",
 "authored/attention_decode_mqa/mqa_decode_padded_cache__m1_aux":"authored/attention_decode_mqa/mqa_decode_padded_cache",
 "authored/attention_decode_mqa/mqa_decode_padded_lengths__m1_aux":"authored/attention_decode_mqa/mqa_decode_padded_lengths",
 "authored/attention_differential/differential_attention__m1_aux":"authored/attention_differential/differential_attention",
 "authored/attention_differential/differential_attention_2__m1_aux":"authored/attention_differential/differential_attention_2",
 "authored/attention_gqa_sink/gqa_sink_sliding_window_attention__m1_aux":"authored/attention_gqa_sink/gqa_sink_sliding_window_attention",
 "authored/attention_local_global/hybrid_local_global_head_attention__m1_aux":"authored/attention_local_global/hybrid_local_global_head_attention",
 "authored/attention_local_global/local_global_head_attention__m1_aux":"authored/attention_local_global/local_global_head_attention",
 "authored/attention_local_global/mixed_local_global_head_attention__m1_aux":"authored/attention_local_global/mixed_local_global_head_attention",
 "authored/attention_sliding_window/gemma3_sliding_window_gqa_attention__m1_aux":"authored/attention_sliding_window/gemma3_sliding_window_gqa_attention",
 "authored/attention_softcap/gemma2_softcap_causal_attention__m1_aux":"authored/attention_softcap/gemma2_softcap_causal_attention",
 "authored/attention_softcap/gemma2_softcap_causal_attention_2__m1_aux":"authored/attention_softcap/gemma2_softcap_causal_attention_2",
 "authored/attention_softcap/gemma2_softcap_sliding_window_attention__m1_aux":"authored/attention_softcap/gemma2_softcap_sliding_window_attention",
 "scraped/maxtext/ragged_attention_v2__m1_batch":"scraped/maxtext/ragged_attention_v2"}
for ch, par in CHILD.items():
    cls, stn, task, fam, why = L[par]
    a(ch, cls, stn, task, fam, "m1 mutation of " + par.split('/')[-1] + " (extra output / split-K / microbatch): " + why)
a("authored/attention_alibi/bloom_alibi_padded_attention_block", "C", "", "5p_Flex_Attention", "val_family", "a VAL problem (dataset-v1 val); its JAXBench relation is family-level (B far)")
a("authored/attention_relative_bias/t5_relative_bias_attention_3", "C", "", "5p_Flex_Attention", "val_family", "a VAL problem; JAXBench relation family-level (B far)")
# ---------------- KernelBench L1 'P_TASK_TWINS' skipped before porting (port_kernelbench.py:27-30) ----------------
T = "kb/l1/"
for st_, cls, stn, task, fam, why in [
 ("1_Square_matrix_multiplication_","A","","8p_GEMM","gemm_epilogue","square GEMM: 8p modulo shape"),
 ("2_Standard_matrix_multiplication_","A","","8p_GEMM","gemm_epilogue","GEMM: 8p modulo shape"),
 ("6_Matmul_with_large_K_dimension_","A","","8p_GEMM","gemm_epilogue","GEMM with large K: 8p modulo shape"),
 ("7_Matmul_with_small_K_dimension_","B","near","8p_GEMM","gemm_epilogue","K=64, 32768x32768 output: output-write-bound, a different regime from 8p"),
 ("8_Matmul_with_irregular_shapes_","A","","8p_GEMM","gemm_epilogue","GEMM: 8p modulo shape"),
 ("9_Tall_skinny_matrix_multiplication_","B","near","8p_GEMM","gemm_epilogue","K=32 outer-product-like GEMM: memory-bound regime"),
 ("13_Matmul_for_symmetric_matrices","A","","8p_GEMM","gemm_epilogue","symmetric inputs, plain GEMM"),
 ("16_Matmul_with_transposed_A","A","","8p_GEMM","gemm_epilogue","GEMM with a transposed operand (layout)"),
 ("17_Matmul_with_transposed_B","A","","8p_GEMM","gemm_epilogue","GEMM with a transposed operand (layout)"),
 ("18_Matmul_with_transposed_both","A","","8p_GEMM","gemm_epilogue","GEMM with transposed operands (layout)"),
 ("36_RMSNorm_","A","","12p_RMSNorm","norm","x/sqrt(mean(x^2,dim=1)+eps): RMSNorm without weight over channels"),
 ("95_CrossEntropyLoss","B","near","13p_Cross_Entropy","cross_entropy","plain CE over given 4096-class logits; 13p's vocab GEMM is absent"),
 ("97_ScaledDotProductAttention","B","near","1p_Flash_Attention","attention","torch SDPA with is_causal=False on given Q/K/V: 1p minus the causal mask"),
]: a(T+st_, cls, stn, task, fam, why)
EOF
echo ok