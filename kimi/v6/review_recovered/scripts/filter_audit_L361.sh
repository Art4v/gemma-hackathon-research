S=/private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad; cat > $S/review/labels.py <<'EOF'
# problem_id -> (class, strength, closest_jaxbench_task, family, justification)
# class: A instance-level (keep excluded) | B family-level (re-admit under instance rule) | C other (val / false positive / not a JAXBench question)
# strength (B only): near = dominant core differs from the task in exactly one respect; far = otherwise
L = {}
def a(pid, cls, st, task, fam, why): L[pid] = (cls, st, task, fam, why)
K = "kernelbook/"
# ---------------- algo review kbk + prod (88) ----------------
for p, why in [
 ("AttentionUnit_13009", "non-causal single-head SDPA over HW=4096 inside InstanceNorm + 1x1-conv projections; 1p is causal MHA on given Q/K/V"),
 ("Attention_17819", "Luong-general cross-attention: non-causal softmax(xK^T)V with K=V=src, plus concat/out-proj/tanh"),
 ("Attention_8907", "Luong-general cross-attention, non-causal, K=V=context, plus concat/out-proj/tanh"),
 ("CrossModalAttention_2752", "nn.MultiheadAttention layer (QKV/out projections), non-causal 4-head SDPA core"),
 ("GlobalNonLocal_17334", "non-local block: 1x1 projections around non-causal single-head SDPA (d=256)"),
 ("MAB_4614", "Set-Transformer MAB: projections + non-causal 4-head SDPA + Q residual + ReLU FC"),
 ("MultiHeadAttention_14276", "MHA layer (QKV/out proj, residual, LayerNorm) around non-causal SDPA"),
 ("MultiHeadAttention_4530", "fused-QKV MHA layer + residual + LayerNorm around non-causal SDPA"),
 ("MultiheadAttention_17488", "bare non-causal MHA SDPA (channel-first layout, no projections): 1p minus the causal mask (= KernelBench L1 #97)"),
 ("NLgate_13812", "bare non-causal single-head unscaled SDPA with a q residual; differs from 1p by mask"),
 ("NonLocal_3178", "1x1-conv QKV + non-causal single-head unscaled SDPA over 4096 positions"),
 ("PAM_Module_15973", "DANet PAM: 1x1-conv QKV + non-causal single-head SDPA + residual"),
 ("PositionalAttentionModule_7559", "DANet position attention: non-causal single-head SDPA, channel-first, residual"),
 ("SAB_4902", "Set attention block: projections + non-causal 4-head SDPA + residual FC"),
 ("SANet_5899", "mean-variance norm + 1x1 convs + non-causal single-head SDPA (d=512)"),
 ("ScaleDotProductAttention_8484", "bare non-causal scaled dot-product attention (no projections): 1p minus the causal mask"),
 ("SelfAttention_1864", "QKV-projected non-causal MHA (seq 2048)"),
 ("SelfAttention_241", "nn.MultiheadAttention: packed QKV GEMM + non-causal 32-head SDPA + out GEMM"),
 ("SelfAttention_2808", "SAGAN self-attention: 1x1 convs + non-causal single-head SDPA (qk 128, v 512)"),
 ("SelfAttention_3012", "Q/K/V linears + non-causal single-head scaled SDPA"),
 ("SelfAttention_naive_6884", "QKV linears + non-causal single-head SDPA, head_dim 1024"),
 ("TransformerLayer_11283", "non-causal 4-head MHA (hd 512) wrapped in QKV/out/fc1/fc2 GEMMs and residuals"),
 ("grouped_token_attention", "QKV linears, 4-token grouping reshape, non-causal SDPA, out linear"),
]: a(K+p, "B", "near", "1p_Flash_Attention", "attention", why)
for p, why in [
 ("DecoderLayer_1082", "full decoder layer (self+cross MHA, 3 LayerNorms, FFN); attention ~34 of ~240 GFLOP"),
 ("ISAB_4635", "Set-Transformer ISAB; projections/FCs ~80% of FLOPs, 128 inducing points"),
 ("ISAB_8416", "two MAB blocks; attention ~5% of FLOPs"),
 ("MAXATTN_17392", "single max-pooled query per batch; no SxS score matrix"),
 ("PAMA_13038", "attention ~1/3 of FLOPs among reflect-pad convs, norms and gated fusion"),
 ("SelfAttAggregate_8134", "softmax(A^T A)A with max+sum pooling dominating bytes and a GEMV output projection"),
 ("SequenceQuantizerSoftEMA_6385", "cosine soft-quantization against a fixed codebook + entropy loss with a global column sum"),
 ("SimpleAttention_14124", "attention pooling: one learned query vector, softmax over sequence, weighted sum (GEMV-shaped, Lq=1)"),
 ("SimpleAttention_60", "attention pooling: matvec score, softmax over sequence, weighted sum"),
 ("conformer_relpos_mhsa", "two-term (content + positional) score with an input mask, non-causal"),
 ("linformer_attention", "Linformer: K/V projected along sequence (2048->256) before softmax; distinct efficient-attention algorithm"),
]: a(K+p, "B", "far", "1p_Flash_Attention", "attention", why)
a("github/wall_attention", "B", "far", "2p_GQA_Attention", "attention", "causal GQA but per-channel log-sigmoid gates inside the dot product + decay bias + sink; not a plain QK^T")
a("github/sam_decomposed_relpos_attention", "B", "far", "5p_Flex_Attention", "attention", "non-causal MHA with a q-dependent decomposed 2D rel-pos bias computed in-kernel + projections; 5p is causal with a loaded dense bias")
a("github/swin_shifted_window_attention", "B", "far", "5p_Flex_Attention", "attention", "Swin: roll + 7x7 window partition, per-window MHSA over 49 tokens with gathered rel-pos bias and shift mask")
a(K+"AttModel_10357", "B", "near", "5p_Flex_Attention", "attention", "single-head non-causal SDPA with a dense multiplicative/additive mask score-mod, ReLU-linear projections")
a(K+"AttModel_13790", "B", "near", "5p_Flex_Attention", "attention", "same as AttModel_10357: non-causal masked single-head SDPA between GEMM+bias+ReLU projections")
a(K+"mixture_synthesizer_attention", "B", "near", "5p_Flex_Attention", "attention", "non-causal single-head softmax(QK^T + learned (L,L) bias)V with projections; 5p is causal MHA with an (H,S,S) bias input")
a(K+"music_transformer_skew_relattn", "B", "far", "5p_Flex_Attention", "attention", "relative logits q@E_r^T realigned by the skew trick (content-dependent), not an added bias tensor")
a(K+"transformerxl_rel_partial_attn", "B", "far", "5p_Flex_Attention", "attention", "Transformer-XL AC+BD with rel_shift, o_net, residual and LayerNorm")
a(K+"SelfAttentionPooling_17848", "B", "far", "6p_Paged_Attention", "attention", "linear H->1 scorer + mask + softmax over T + weighted sum (pooling, Lq=1, no paging/GQA)")
a(K+"SelfAttn_3792", "B", "far", "6p_Paged_Attention", "attention", "linear D->1 scorer + softmax over sequence + weighted sum (pooling)")
a(K+"FRN_self_17244", "A", "", "12p_RMSNorm", "norm", "x*rsqrt(mean(x^2 over trailing HW)+|eps|)*gamma+beta = RMSNorm over the contiguous last axis plus a bias")
a(K+"PixelNorm_4480", "A", "", "12p_RMSNorm", "norm", "x*rsqrt(mean(x^2)+eps): RMSNorm without the weight; only the reduction axis (layout) differs")
for p, st, why in [
 ("AttentionCrossEntropy_8444", "near", "dense soft-target CE over given logits; 13p's dominant stage (8192x4096x128256 vocab GEMM) is absent"),
 ("CELoss_13876", "near", "argmax labels + log_softmax CE over 128 classes, no vocab GEMM (= KernelBench L1 #95 shape of problem)"),
 ("CrossEntropyLossOneHot_5009", "near", "log_softmax CE with dense one-hot labels over given logits; no vocab GEMM"),
 ("CrossEntropyLossTF_302", "far", "broadcast makes it sum -log_softmax over classes at spatial positions; not target-logit CE; no GEMM"),
 ("CrossEntropy_3310", "near", "plain CE over given logits (class axis 1); no vocab GEMM"),
 ("FocalLoss_13061", "near", "focal loss with gamma=0 = plain CE over given logits; no vocab GEMM"),
 ("SoftCrossEntropy_16792", "near", "soft-target CE over the channel axis; no integer gather, no vocab GEMM"),
 ("SoftTargetCrossEntropy_10953", "near", "soft-target CE over 512 classes; no vocab GEMM"),
 ("SoftTargetCrossEntropy_9484", "near", "soft-target CE over 128 classes; no vocab GEMM"),
]: a(K+p, "B", st, "13p_Cross_Entropy", "cross_entropy", why)
a(K+"OuterProductLayer_1919", "B", "near", "14p_Ragged_Dot", "grouped_matmul", "uniform batched einsum wrapped in a triu pair-gather prologue and a dot-with-q epilogue that collapses the output")
a("github/based_chunk", "B", "far", "15p_RetNet_Retention", "linear_attn_ssm", "Based: 2nd-order Taylor feature map, chunked prefix states of size dk^2 x dv, no decay")
a("github/hedgehog_linear_attention", "B", "near", "15p_RetNet_Retention", "linear_attn_ssm", "causal chunked linear attention with a learned softmax feature map and no decay (gamma=1); 15p is per-head-decayed retention with abs-sum normalisation")
a(K+"FullyConnected_7520", "A", "", "20k_Gemm_Multiply_LeakyReLU", "gemm_epilogue", "GEMM + LeakyReLU(0.1): 20k's chain minus the bias and the x2 constant")
a(K+"FullyConnected_8120", "A", "", "20k_Gemm_Multiply_LeakyReLU", "gemm_epilogue", "GEMM + LeakyReLU(0.1): 20k's chain minus bias and constant")
a(K+"EnsembleModel_7391", "A", "", "37k_Matmul_Swish_Scaling", "gemm_epilogue", "6 of 7 layers (>2/3 FLOPs) are bmm+bias+SiLU = 37k's chain modulo the x2 constant and an ensemble batch axis (borderline: batched)")
a(K+"DoubleDense_362", "A", "", "41k_Gemm_Add_ReLU", "gemm_epilogue", "two 8192x4096x4096 GEMM+bias+ReLU layers (8/9 of FLOPs) are exactly 41k's chain")
a(K+"MLP_8357", "A", "", "41k_Gemm_Add_ReLU", "gemm_epilogue", "512->1024->1024->512: the two GEMM+bias+ReLU layers are 75% of FLOPs, exactly 41k's chain")
a(K+"ResnetBlockFC_9798", "B", "near", "41k_Gemm_Add_ReLU", "mlp", "pre-activation residual MLP: GEMM+bias+ReLU stage is 50% of FLOPs, second stage GEMM+bias+input residual")
a(K+"mlp_5layer_7259", "A", "", "41k_Gemm_Add_ReLU", "gemm_epilogue", "~95% of FLOPs is one 2048x16384x256 GEMM+bias+ReLU (41k's chain)")
a(K+"MLP_17260", "B", "near", "45k_Gemm_GroupNorm_Swish_Multiply_Swish", "mlp", "GEMM -> LayerNorm -> SiLU -> GEMM: 1-group norm instead of 256-group GN, no multiply/second swish, plus a second GEMM")
a(K+"SinkhornDistance_16855", "B", "far", "56p_DeepSeek_V4_MHC", "mhc_sinkhorn", "log-domain optimal-transport Sinkhorn loss (100 iters on a cost matrix); 56p runs prob-domain Sinkhorn on tiny mixing matrices inside a hyper-connection op")
a(K+"NTXent_6903", "A", "", "64p_Linear_Softmax_Cross_Entropy", "cross_entropy", "normalised z1@z2^T * exp(t), logsumexp minus the diagonal = linear-softmax-CE with W=z2^T and arange labels")
a(K+"SplitCrossEntropyLoss_9282", "A", "", "64p_Linear_Softmax_Cross_Entropy", "cross_entropy", "hiddens@W+b -> logsumexp -> target-logit gather: 64p's fused linear CE with a trivial per-token transform")
a(K+"AR_12699", "B", "near", "8p_GEMM", "gemm_epilogue", "x@w+b with out_features=1: a bandwidth-bound GEMV, a different kernel regime (KernelBench lists matvec separately, L1 #4)")
for p, why in [
 ("Classifier_18035", "concat + one 8192x4096x1024 GEMM + bias"),
 ("Decoder_2926", "one shared x@W+b across a channel axis + swapaxes"),
 ("DisplacementPrediction_3602", "data@w+b (M=262144, K=N=256)"),
 ("LearnedUpsampling1d_3140", "ConvTranspose1d with stride=kernel = one GEMM + bias + reshape"),
 ("LearnedUpsampling1d_7764", "ConvTranspose1d with stride=kernel = one GEMM + bias + interleave"),
 ("LinearWithChannel_10279", "x shared across C weight slices = one dense GEMM + bias in (C,B,O) layout"),
 ("QuantPointwiseConv_2886", "identity quant stubs; x@w+b"),
]: a(K+p, "A", "", "8p_GEMM", "gemm_epilogue", why + ": a plain GEMM with at most a bias")
a(K+"Generator_13017", "B", "near", "8p_GEMM", "mlp", "three GEMMs with ELU / sigmoid / none epilogues; no k-task chain, no single dominant stage")
a(K+"MLP_12940", "B", "far", "8p_GEMM", "mlp", "tiny 128->40->40->4 tanh MLP: a memory-bound fused per-row chain")
a(K+"ResnetBlockFC_2418", "B", "near", "8p_GEMM", "mlp", "pre-activation residual MLP (two equal GEMMs, ReLU before each, input residual)")
a(K+"AFTSimple_2134", "B", "far", "9p_SwiGLU_MLP", "mlp", "AFT-simple: three projections, sigmoid(q)*softmax_features(k)*v, down projection")
for p, why in [
 ("CMlp_11655", "ungated pointwise GELU MLP (two 1x1x1 convs); 9p is gated SwiGLU with three GEMMs"),
 ("FFN_8077", "ungated ReLU FFN 1024->4096->1024 + mask multiply; the 41k-like layer is 50% of FLOPs"),
 ("FeedForwardNetwork_5615", "ungated tanh-GELU MLP"),
 ("GatedDense_15172", "relu(xWh+b)*sigmoid(xWg+b): 9p's dual-GEMM gate stage with other activations and no down projection"),
 ("ResnetBlockFC_7838", "pre-ReLU two-layer MLP + identity residual"),
 ("SimpleMLPGen_with_meta_feature_13166", "ungated Linear->ELU->Linear MLP"),
]: a(K+p, "B", "near", "9p_SwiGLU_MLP", "mlp", why)
a(K+"FeedForward_7227", "B", "far", "9p_SwiGLU_MLP", "mlp", "ungated FFN wrapped in add+LayerNorm before and residual+LayerNorm after")
# ---------------- algo review kbkc (25) ----------------
a(K+"LabelSmoothSoftmaxCEV1_6443", "B", "near", "13p_Cross_Entropy", "cross_entropy", "label-smoothed CE with ignore-index over given logits; no vocab GEMM")
a(K+"StandardNLL_14131", "B", "far", "13p_Cross_Entropy", "cross_entropy", "NLL gather of given log-probs + sequence-mask mean; no softmax, no GEMM")
a(K+"Attention_8960", "B", "far", "1p_Flash_Attention", "attention", "attention pooling: one learned query, softmax over sequence, weighted sum (Lq=1, K=V=x)")
a(K+"CAM_Module_4021", "B", "far", "1p_Flash_Attention", "attention", "DANet channel attention: Gram matrix over 256 channel tokens x 16384 dims, negated scores, Q=K=V")
a(K+"ISAB_16957", "B", "far", "1p_Flash_Attention", "attention", "two set-attention blocks; SDPA ~47% of FLOPs next to bias GEMMs")
a(K+"ISAB_4910", "B", "far", "1p_Flash_Attention", "attention", "two MAB blocks; cross-attention ~20% of FLOPs")
a(K+"MultiHeadedAttention_15199", "B", "near", "1p_Flash_Attention", "attention", "Q/K/V/out projections around non-causal single-head (d=1024) SDPA")
a(K+"NonLocalBlock2D_13181", "B", "far", "1p_Flash_Attention", "attention", "softmax over the query axis (column normalisation): no online softmax over keys")
a(K+"NonLocalLayer_17138", "B", "far", "1p_Flash_Attention", "attention", "projections + non-causal SDPA ~44% of FLOPs + residual")
a(K+"SAB_16960", "B", "near", "1p_Flash_Attention", "attention", "set attention block: projections + non-causal MHA SDPA + residual FC")
a(K+"SAB_8410", "B", "far", "1p_Flash_Attention", "attention", "set attention block; SDPA ~45% of FLOPs")
a(K+"SANet_8218", "B", "near", "1p_Flash_Attention", "attention", "norm + 1x1 convs around non-causal single-head SDPA (seq 4096, d 1024)")
a(K+"SelfAttentionGPT2_9334", "B", "near", "1p_Flash_Attention", "attention", "fused QKV + 8-head SDPA with NO causal mask + out projection (reference code has no mask)")
a(K+"VisionLanguageFusionModule_16722", "B", "near", "1p_Flash_Attention", "attention", "packed-QKV non-causal cross-attention (L=1024,S=2048) + out linear + elementwise multiply")
a(K+"FC_1124", "A", "", "20k_Gemm_Multiply_LeakyReLU", "gemm_epilogue", "one GEMM (no bias) + LeakyReLU(0.2): 20k's chain modulo constants")
a(K+"Feedforward_9337", "A", "", "41k_Gemm_Add_ReLU", "gemm_epilogue", "99.8% of FLOPs a 131072x512x512 GEMM+bias+ReLU; tiny 512->1 head")
a(K+"MLP_16942", "B", "near", "41k_Gemm_Add_ReLU", "mlp", "two equal 2048x2048 layers: GEMM+bias+ReLU (50%) then GEMM+bias")
a(K+"MLP_4026", "A", "", "41k_Gemm_Add_ReLU", "gemm_epilogue", "two GEMM+bias+ReLU layers are >99% of FLOPs (the 640->512 one alone is 2/3); tiny sigmoid head")
a(K+"SetConv_1791", "B", "far", "41k_Gemm_Add_ReLU", "mlp", "four set encoders; masked mean pooling over the set axis dominates bytes")
a(K+"TwoMLPHead_17912", "A", "", "41k_Gemm_Add_ReLU", "gemm_epilogue", "first layer (~80% FLOPs) is a 2048x8192x2048 GEMM+bias+ReLU, second the same chain")
a(K+"DGN_10362", "B", "far", "5p_Flex_Attention", "attention", "two masked single-head attentions (~30% FLOPs) among 10 GEMM+bias+ReLU layers")
a(K+"MultiHeadAttention_15184", "B", "near", "5p_Flex_Attention", "attention", "projections + non-causal MHA with a dense additive mask; 5p is causal with an (H,S,S) bias")
a(K+"AR_9990", "B", "near", "8p_GEMM", "gemm_epilogue", "Linear(128->1)+bias over 262k rows: bandwidth-bound GEMV")
a(K+"SelfAttention_5795", "B", "near", "8p_GEMM", "gemm_epilogue", "two bias projections + batched Q@K^T/sqrt(d): a chain of GEMMs, no softmax/V")
a(K+"MixerMLP_13064", "B", "far", "9p_SwiGLU_MLP", "mlp", "LayerNorm prologue + ungated GELU MLP")
# ---------------- algo review kbkd (28) ----------------
a(K+"L2Norm_13237", "A", "", "12p_RMSNorm", "norm", "x/(sqrt(sum_c x^2)+eps)*w: RMSNorm up to the constant sqrt(C), eps placement and reduction axis")
a(K+"L2Norm_5161", "A", "", "12p_RMSNorm", "norm", "same computation as L2Norm_13237 at another size")
a(K+"partCE_8260", "B", "near", "13p_Cross_Entropy", "cross_entropy", "per-pixel CE over 128 classes with masked-argmax labels; no vocab GEMM")
a(K+"Attention_17195", "B", "far", "1p_Flash_Attention", "attention", "single-query attention pooling (learned query, K=V=x)")
a(K+"CPAMDec_10410", "B", "near", "1p_Flash_Attention", "attention", "non-causal single-head cross-attention (4096 pixels x 256 centers) + residual")
a(K+"Bi_Attention_17614", "B", "far", "1p_Flash_Attention", "attention", "co-attention: one score matrix, two masked softmaxes over each axis")
a(K+"InterModalityUpdate_17913", "B", "far", "1p_Flash_Attention", "attention", "two cross-attentions (~45% FLOPs) + Linear+ReLU projections")
a(K+"Attention_15176", "B", "near", "1p_Flash_Attention", "attention", "1x1-conv projections around non-causal single-head SDPA (~90% FLOPs)")
a(K+"_ChannelAttentionModule_5287", "B", "far", "1p_Flash_Attention", "attention", "DANet channel Gram attention (512 channel tokens, dim 4096, negated scores)")
a(K+"MultiHeadAttention_7344", "B", "near", "1p_Flash_Attention", "attention", "4-head non-causal SDPA with projected key reused as value, concat + Linear + tanh")
a(K+"GlobalAttentionGeneral_5169", "B", "near", "1p_Flash_Attention", "attention", "non-causal single-head cross-attention (4096 queries x 128 keys), K=V")
a(K+"SelfAttention_10797", "B", "near", "1p_Flash_Attention", "attention", "non-causal 8-head MHA with projections")
a(K+"Multihead_Attention_Layer_910", "B", "near", "1p_Flash_Attention", "attention", "1x1-conv QKV + non-causal 4-head SDPA + out conv")
a(K+"PAM_Module_4494", "B", "near", "1p_Flash_Attention", "attention", "non-causal single-head unscaled SDPA (d=512) + 1x1 projections + residual")
a(K+"_MLP_C_5185", "A", "", "20k_Gemm_Multiply_LeakyReLU", "gemm_epilogue", "99.6% of FLOPs is GEMM+bias+LeakyReLU (20k's chain minus the x2), tiny 512->2 head")
a(K+"NeuralNetMultiplePositionalArgumentsMultiOutputsWithoutDependency_5906", "B", "near", "38k_Matmul_Dropout_Softmax", "gemm_epilogue", "Linear+bias+softmax but the softmax runs over a strided non-feature axis (different reduction)")
a(K+"MLP_8911", "A", "", "41k_Gemm_Add_ReLU", "gemm_epilogue", "first layer (~86% FLOPs) is exactly GEMM+bias+ReLU")
a(K+"Generator_2161", "B", "near", "41k_Gemm_Add_ReLU", "mlp", "three equal 512x512 layers with ELU / sigmoid / none")
a(K+"MLPNet_775", "A", "", "41k_Gemm_Add_ReLU", "gemm_epilogue", "784->500 GEMM+bias+ReLU ~75% of FLOPs; second layer same chain")
a(K+"Attention_16802", "A", "", "41k_Gemm_Add_ReLU", "gemm_epilogue", "~94% of FLOPs is one GEMM+bias+ReLU identical to 41k plus a broadcast multiply (borderline)")
a(K+"TwoMLPHead_2335", "A", "", "41k_Gemm_Add_ReLU", "gemm_epilogue", "first layer ~92% of FLOPs is exactly GEMM+bias+ReLU")
a(K+"MLP_16034", "A", "", "41k_Gemm_Add_ReLU", "gemm_epilogue", "first layer ~85% of FLOPs is exactly GEMM+bias+ReLU")
a(K+"MLPNetwork_4815", "A", "", "41k_Gemm_Add_ReLU", "gemm_epilogue", "three equal 512x512 layers, two are GEMM+bias+ReLU (exactly 2/3 of FLOPs; borderline)")
a(K+"MHSA_12172", "B", "far", "5p_Flex_Attention", "attention", "BoTNet MHSA: content-position term q.rel added to scores, 2D non-causal")
a(K+"CRF_14013", "B", "far", "8p_GEMM", "gemm_epilogue", "small GEMM+bias then a 128x broadcast-add expansion that dominates bytes")
a(K+"GEGLU_11635", "B", "near", "9p_SwiGLU_MLP", "mlp", "GEGLU gate/up dual GEMM with GELU, no down projection")
a(K+"HighwayMLP_10676", "B", "far", "9p_SwiGLU_MLP", "mlp", "two GEMMs with ReLU x softmax-over-batch gate + highway; no down projection")
a(K+"TransformerEncoderFeedForward_17858", "B", "far", "9p_SwiGLU_MLP", "mlp", "ungated FFN with residual + LayerNorm epilogue")
EOF
echo ok