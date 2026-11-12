import json
out={
 "verified_instance_in_pool_not_in_audit": {
  "9p_SwiGLU_MLP": ["authored/swiglu_variant_geglu/geglu_mlp_fused_gate_up","authored/swiglu_variant_geglu/geglu_fused_gate_up_mlp","authored/swiglu_variant_geglu/geglu_fused_mlp","authored/moe_shared_plus_routed/qwen_moe_shared_routed_combine","authored/moe_shared_plus_routed/qwen_moe_shared_routed_combine_2","authored/moe_shared_plus_routed/qwen_moe_shared_plus_routed_combine"],
  "12p_RMSNorm": ["kernelbook/PixelNorm_16870","kernelbook/PixelNorm_18399","kernelbook/PixelNorm_890","kernelbook/L2Norm_7890","kernelbook/L2Norm_27","kernelbook/L2Norm_6153","kernelbook/L2Normalization_17502","kernelbook/ScaledL2Norm_6171","kb/l1/39_L2Norm_","kb/l1/39_L2Norm___bf16","kb/l1/39_L2Norm___m1_ragged"],
  "18k_Conv2D_ReLU_BiasAdd": ["kernelbook/SqueezeInitBlock_6634","kernelbook/RC_6332"],
  "41k_Gemm_Add_ReLU": ["kernelbook/ConvEncoder_8588"],
  "20k_Gemm_Multiply_LeakyReLU": ["kernelbook/TinyDiscriminator_13761","kernelbook/GraphConvolution_8391"],
  "38k_Matmul_Dropout_Softmax": ["kernelbook/GumbelSoftmax_9252"],
  "8p_GEMM": ["kernelbook/Fusion3_MinusFCLayer_17850","kernelbook/Fusion2_MinusFCLayer_17842","kernelbook/SkipModule_16307","kernelbook/Coboundary_8862"],
  "57p_SparseCore_Ragged_Gather": ["kernelbook/ItemInferenceNetwork_12467","kernelbook/Join_17327"]},
 "unsure_in_pool_not_in_audit": {
  "13p/64p": ["authored/softcapped_logits/gemma_softcapped_logits_logsoftmax","authored/softcapped_logits/gemma_softcapped_logits_logsoftmax_2","authored/softcapped_logits/gemma_softcapped_logits_logsoftmax__m1_splitk","authored/softcapped_logits/gemma_softcapped_logits_logsoftmax_2__m1_splitk"],
  "12p": ["authored/norm_gated/gated_rmsnorm_per_head"],
  "38k/10p router": ["authored/moe_topk_routing/moe_router_softmax_topk_renorm","authored/moe_topk_routing/moe_router_softmax_topk_renorm__m1_splitk"]},
 "relu_stacks_meeting_audit_A_standard": json.load(open('pool_relu_stacks.json')),
}
json.dump(out,open('pool_contamination.json','w'),indent=1)
print(sum(len(v) for v in out['verified_instance_in_pool_not_in_audit'].values()), sum(len(v) for v in out['unsure_in_pool_not_in_audit'].values()))
