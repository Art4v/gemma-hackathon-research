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
# ---------------- filter_extra / opchain / neardup (karthik/kbk kbk+kbkd, aarav/kbk-c kbkc) not already labelled ----------------
VALPOOL = "C"
def c_val(p, task, why): a(K+p, "C", "", task, "val_family", why)
for p in ["MaxPoolStride1_9509","MaxPoolPad_542","NasMaxPoolBlock_17684","MaxPoolStride1_11689","MaxPool2d_6395","MaxPool_17153",
          "SampaddingMaxPool1D_6195","MaxPoolPad_17152","MaxPool_3293","MaxPool_12803","MaxPoolStride1_2354","MyMaxPool1dPadSame_2963",
          "MaxPool_5559","MyMaxPool1dPadSame_6833","MaxPoolPad_9868","IdentityPadding_1710","MaxPool2dDynamicSamePadding_11436"]:
    c_val(p, "val:kb/l1/43_Max_Pooling_3D", "max pooling, removed for op-multiset Jaccard 1.00 with the val problem kb/l1/43; JAXBench has no pooling-only task")
c_val("MSE_3682", "val:kb/l1/94_MSELoss", "mean((pred-gt)^2) is IDENTICAL to the val problem kb/l1/94_MSELoss (val instance duplicate; currently in the Kimi pool)")
c_val("FastSigmoid_13172", "val:kb/l1/30_Softsign", "x/(|x|+1) is IDENTICAL to the val problem kb/l1/30_Softsign (val instance duplicate; currently in the Kimi pool)")
for p in ["RMSELoss_471","MulticlassSegmentationLoss_5714","SuperLoss_6372","MaskedMSE_11077"]:
    c_val(p, "val:kb/l1/94_MSELoss", "MSE-family loss (val family, not identical); no JAXBench relation")
c_val("GeneralRelu_756", "val:kb/l1/43_Max_Pooling_3D", "elementwise leaky-ReLU/sub/clamp; Jaccard 1.00 with max-pool is a multiset artifact (false positive)")
c_val("GeneralRelu_17430", "val:kb/l1/43_Max_Pooling_3D", "elementwise leaky-ReLU/sub/clamp; multiset false positive vs max-pool")
c_val("PatchApplier_5711", "val:scraped/tpu-inference/spmm", "loop of jnp.where overlays; false positive vs the val spmm problem")
a(K+"equalized_linear_7234", "A", "", "8p_GEMM", "gemm_epilogue", "(x*scale)@w+bias: plain GEMM with a constant scale and bias (removed by a val rule, but an 8p instance; currently in the Kimi pool)")
a(K+"EmbedNoise_9273", "A", "", "20k_Gemm_Multiply_LeakyReLU", "gemm_epilogue", "z@w+b -> LeakyReLU -> reshape: 20k's chain minus the x2 constant (currently in the Kimi pool)")
a(K+"SoftmaxAffineLayer_4235", "B", "near", "38k_Matmul_Dropout_Softmax", "gemm_epilogue", "channel GEMM + bias, /t, LOG_softmax over output channels (38k uses softmax)")
a(K+"SoftmaxAffineLayer_6691", "B", "near", "38k_Matmul_Dropout_Softmax", "gemm_epilogue", "channel GEMM + bias + log_softmax over the output-channel axis (filter named 42k)")
a(K+"Attn_15410", "B", "far", "38k_Matmul_Dropout_Softmax", "attention", "Luong-general scores (enc@W+b)·hidden, softmax over sequence; returns weights only")
a(K+"SelfAttnPooler_12096", "B", "far", "38k_Matmul_Dropout_Softmax", "attention", "masked attention pooling (linear scorer, softmax over T, weighted sum)")
a(K+"Attn_10095", "B", "far", "38k_Matmul_Dropout_Softmax", "attention", "Bahdanau additive attention scores: tanh(cat@W+b)·v, softmax")
a(K+"Attention_8837", "B", "far", "38k_Matmul_Dropout_Softmax", "attention", "additive attention with ReLU energy, softmax over time")
a(K+"Encoder_attention_17177", "B", "far", "38k_Matmul_Dropout_Softmax", "attention", "attention pooling: linear scorer, softmax, weighted sum")
a(K+"AttPool_18021", "B", "far", "38k_Matmul_Dropout_Softmax", "attention", "attention pooling: softmax(x@w+b) weighted sum")
a(K+"SelfAttn_692", "B", "far", "38k_Matmul_Dropout_Softmax", "attention", "attention pooling over keys/values with a linear scorer")
a(K+"SimpleCNN32Filter_12057", "B", "near", "18k_Conv2D_ReLU_BiasAdd", "conv", "conv(k10,s2)+bias then ReLU (18k adds bias after ReLU) + an FC layer")
a(K+"DepthwiseSeparableConv_8314", "B", "near", "18k_Conv2D_ReLU_BiasAdd", "conv", "depthwise conv1d + pointwise GEMM+bias+ReLU (stencil prologue + 41k-like stage)")
a(K+"convnet_3957", "B", "near", "18k_Conv2D_ReLU_BiasAdd", "conv", "two convs (conv+bias+ReLU, conv+bias)")
a(K+"Encoder1_14018", "B", "near", "18k_Conv2D_ReLU_BiasAdd", "conv", "conv+bias, reflect pad, conv+bias+ReLU")
a(K+"CAModel_14148", "B", "near", "18k_Conv2D_ReLU_BiasAdd", "conv", "two convs (conv+bias+ReLU, conv+bias)")
for p, why in [("ChanNorm_2745","LayerNorm over channels"),("ChanNorm_5651","LayerNorm over channels (std+eps)"),("LayerNorm_6248","LayerNorm (ddof=1) over an axis"),
               ("LayerNorm_4866","per-sample LayerNorm over all non-batch axes"),("LayerNorm_4861","per-sample LayerNorm"),("VirtualBatchNorm_15446","batch-statistics BatchNorm"),
               ("LayerNorm1D_9941","LayerNorm over axis 1"),("LayerNormChan_10748","LayerNorm over channels"),("LayerNorm_7669","per-sample LayerNorm"),
               ("LayerNorm_3098","LayerNorm over an axis"),("Normalize_4644","centred normalisation with gain/bias"),("MyBatchNorm_4319","BatchNorm over a stat axis"),
               ("LayerNorm_78","per-sample LayerNorm"),("LayerNormalization_18037","LayerNorm over last axis"),("LayerNorm_17385","LayerNorm over last axis"),
               ("Fp32GroupNorm_9219","GroupNorm alone"),("TestTimeIN_8582","InstanceNorm + clamp without a conv")]:
    a(K+p, "B", "far", "22k_Conv2d_InstanceNorm_Divide", "norm", why + " only; 22k's dominant stage is a 3x3 conv and the nearest standalone norm task (12p) is uncentred")
for p, why in [("Conv2dWS_15083","weight-standardised conv (norm on weights, not activations)"),("StdConv2d_6818","weight-standardised conv"),
               ("StdConv3d_9599","weight-standardised 3D conv"),("StdConv2d_5297","weight-standardised grouped conv")]:
    a(K+p, "B", "far", "22k_Conv2d_InstanceNorm_Divide", "conv", why + "; 22k normalises the conv OUTPUT per instance")
a(K+"ConvLayer_17320", "A", "", "22k_Conv2d_InstanceNorm_Divide", "conv", "reflect-pad + conv + bias + InstanceNorm + affine: 22k's chain modulo padding/stride and the /2 constant")
a(K+"PositionEmbedder_11333", "C", "", "57p_SparseCore_Ragged_Gather", "other", "x + table[arange(S)] is a broadcast add of a contiguous slice; no ragged gather (false positive)")
a(K+"LearnableMax_2764", "C", "", "14p_Ragged_Dot", "other", "two elementwise maxima with parameters; text near-dup artefact (false positive)")
a(K+"AttentionalColorizedListenerDecoder_8896", "C", "", "13p_Cross_Entropy", "other", "Mahalanobis-style scores + softmax; no cross-entropy (false positive)")
a(K+"AlphaMish_3991", "B", "far", "27k_Matmul_Mish_Mish", "elementwise", "elementwise Mish-variant activation only: a fragment of 27k's epilogue without its GEMM")
a(K+"Discriminator_14996", "B", "far", "36k_Matmul_Sigmoid_Sum", "gemm_epilogue", "bilinear DGI discriminator (node@W dotted with summary) + sigmoid")
a(K+"Decoder_12947", "B", "far", "30k_Matmul_Scaling_ResidualAdd", "gemm_epilogue", "mask multiply + basis einsum + overlap-and-add scatter (multiset Jaccard 1.00 is an artefact)")
a(K+"MemoryMoCo_15724", "B", "near", "21k_Gemm_Divide_Sum_Scaling", "gemm_epilogue", "q@memory^T + row dot + threshold + concat + /T; 21k sums over features")
a(K+"Attention_2381", "B", "near", "1p_Flash_Attention", "attention", "bare non-causal SDPA with K=V=keys (no projections, no scale)")
a(K+"GlobalAttentionGeneral_8369", "B", "near", "1p_Flash_Attention", "attention", "non-causal cross-attention (image positions x context), no projections")
a(K+"ParameterOutput_11424", "B", "near", "38k_Matmul_Dropout_Softmax", "gemm_epilogue", "GEMM+bias+softmax over axis 1 of a 4D tensor (not the feature axis) + affine")
a(K+"LinearAttention_820", "B", "far", "38k_Matmul_Dropout_Softmax", "attention", "attention pooling softmax(x@w+b) weighted sum")
a(K+"TokenLabelSoftTargetCrossEntropy_2733", "B", "near", "13p_Cross_Entropy", "cross_entropy", "soft-target CE over given logits; no vocab GEMM")
a(K+"CrossEntropy_497", "B", "near", "13p_Cross_Entropy", "cross_entropy", "dense-target CE over class axis 1; no vocab GEMM")
a(K+"SoftTargetCrossEntropy_10360", "B", "near", "13p_Cross_Entropy", "cross_entropy", "weighted soft-target CE over given logits; no vocab GEMM")
for p in ["PixelNorm_9401","PixelNorm_17064","PixelNorm_4790"]:
    a(K+p, "A", "", "12p_RMSNorm", "norm", "x*rsqrt(mean(x^2)+eps) over channels: RMSNorm without the weight (layout differs only)")
a(K+"GCN_14619", "B", "far", "41k_Gemm_Add_ReLU", "gemm_epilogue", "GEMM+bias+residual+ReLU then a second contraction with the adjacency")
a(K+"DQN_hot1_17174", "B", "far", "9p_SwiGLU_MLP", "mlp", "16384->15->128 MLP: tiny hidden width, memory-bound")
a(K+"kAttentionPooling_7782", "B", "far", "38k_Matmul_Dropout_Softmax", "attention", "k-head attention pooling (scores x@theta, softmax, einsum)")
a(K+"Attention_18078", "B", "far", "38k_Matmul_Dropout_Softmax", "attention", "additive attention tanh(cat@W)@v + mask, softmax")
a(K+"SelfAttentionPooling_1698", "B", "far", "38k_Matmul_Dropout_Softmax", "attention", "attention pooling")
a(K+"Linear_1757", "A", "", "8p_GEMM", "gemm_epilogue", "src@W + bias: plain GEMM with a bias")
a(K+"LinModel_17578", "A", "", "38k_Matmul_Dropout_Softmax", "gemm_epilogue", "x@w+b, softmax over the feature axis (-1): exactly 38k's chain")
a(K+"Attention_layer_13080", "A", "", "38k_Matmul_Dropout_Softmax", "gemm_epilogue", "GEMM+bias+softmax over features then an elementwise gate by the input (trivial epilogue; borderline)")
a(K+"perceptron_506", "B", "near", "38k_Matmul_Dropout_Softmax", "gemm_epilogue", "GEMM(256->10)+bias with softmax over axis 1 (a spatial axis, not features)")
a(K+"Hidden2DiscreteDeal_3790", "B", "near", "38k_Matmul_Dropout_Softmax", "gemm_epilogue", "GEMM+bias -> log_softmax, returns logits and log-probs (filter named 44k)")
a(K+"NetEnd_17238", "B", "near", "38k_Matmul_Dropout_Softmax", "gemm_epilogue", "1x1 conv 21->4 channels + bias + softmax: GEMM with K=21,N=4 is memory-bound (different regime)")
a(K+"TemporalDecayRegression_3216", "B", "near", "8p_GEMM", "gemm_epilogue", "exp(-relu(x@w)): GEMM with a non-k epilogue")
a(K+"LinearBlock_7634", "A", "", "41k_Gemm_Add_ReLU", "gemm_epilogue", "x@w+b then max(.,0): exactly 41k's chain")
a(K+"LinearBlock_67", "A", "", "41k_Gemm_Add_ReLU", "gemm_epilogue", "relu(x@w+b): exactly 41k's chain")
a(K+"LinearEncoder_2787", "B", "near", "41k_Gemm_Add_ReLU", "gemm_epilogue", "GEMM+bias+ReLU followed by a LayerNorm epilogue (full-row reduction)")
a(K+"NeuralNetwork_12059", "B", "far", "1p_Flash_Attention", "mlp", "2-layer LeakyReLU MLP + softmax over the BATCH axis; no attention (filter's 1p hit is an artefact)")
a(K+"DenseSAGEConv_2033", "B", "far", "42k_Gemm_Max_Subtract_GELU", "gemm_epilogue", "graph SAGE conv: adj@x, degree normalisation, two GEMMs")
a(K+"UpConcat2d_7560", "A", "", "41k_Gemm_Add_ReLU", "gemm_epilogue", "stride=kernel conv-transpose as one einsum GEMM + bias + ReLU, then crop/concat copy (op chain equal; borderline)")
a(K+"Biaffine_5538", "B", "near", "14p_Ragged_Dot", "gemm_epilogue", "two chained contractions (input1@W then batched einsum with input2); chain equality is an artefact of repeat-collapsing")
a(K+"TextProcessor_11766", "A", "", "41k_Gemm_Add_ReLU", "gemm_epilogue", "reshape, x@w+b, ReLU: exactly 41k's chain")
a(K+"Classifier_1798", "B", "near", "37k_Matmul_Swish_Scaling", "gemm_epilogue", "GEMV (N=1)+bias -> sigmoid -> x mask: sigmoid gates a mask, not the GEMM output (different role)")
a(K+"Biaffine_9240", "B", "far", "14p_Ragged_Dot", "gemm_epilogue", "3-way bilinear einsum 'bxi,oij,byj->boxy'")
a(K+"PositionalEncoding_6197", "A", "", "41k_Gemm_Add_ReLU", "gemm_epilogue", "concat a position column, GEMM+bias+ReLU: 41k's chain with a trivial prologue")
a(K+"NumPredictor_18261", "A", "", "41k_Gemm_Add_ReLU", "gemm_epilogue", "relu(x@w+b): exactly 41k's chain")
a(K+"GLU_1348", "B", "near", "37k_Matmul_Swish_Scaling", "gemm_epilogue", "lin=x@w+b times sigmoid(x): the gate is the INPUT, not the GEMM output (37k is y*sigmoid(y))")
a(K+"LinearEmbedder_13109", "B", "near", "8p_GEMM", "gemm_epilogue", "GEMM+bias then row L2 normalisation (filter named 42k)")
a(K+"NeuralNetPartialNoGradModel_3264", "B", "near", "41k_Gemm_Add_ReLU", "mlp", "two equal 1024x1024 layers: GEMM+bias+ReLU (50%) then GEMM+bias (filter named 9p)")
a(K+"Sentence_Maxpool_14501", "B", "far", "42k_Gemm_Max_Subtract_GELU", "gemm_epilogue", "GEMM+bias, max over the sequence axis, ReLU; 42k maxes over features then subtracts a mean and applies GELU")
a(K+"MetaBilinear_711", "B", "far", "14p_Ragged_Dot", "gemm_epilogue", "bilinear form x1 W x2 + b")
a(K+"MetaBilinear_14504", "B", "far", "14p_Ragged_Dot", "gemm_epilogue", "bilinear form x1 W x2 + b")
# ---------------- dataset-v1 neardup / review (gemma-dataset-claude) ----------------
a("kb/l2/68_Matmul_Min_Subtract", "B", "far", "42k_Gemm_Max_Subtract_GELU", "gemm_epilogue", "GEMM, elementwise min with a constant, subtract a constant; 42k reduces max over features, subtracts the mean, GELU")
a("kb/l2/70_Gemm_Sigmoid_Scaling_ResidualAdd", "B", "near", "30k_Matmul_Scaling_ResidualAdd", "gemm_epilogue", "30k's chain plus a sigmoid (filter named 44k)")
a("kb/l2/51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd", "B", "far", "42k_Gemm_Max_Subtract_GELU", "gemm_epilogue", "different KernelBench L2 problem: subtract, global avg pool, logsumexp, GELU, residual")
a("kb/l2/15_ConvTranspose3d_BatchNorm_Subtract", "B", "far", "25k_Conv3d_GroupNorm_Mean", "conv", "conv-transpose3d + BatchNorm + subtract spatial mean vs conv3d + GroupNorm + mean")
a("kb/l2/4_Conv2d_Mish_Mish", "B", "near", "27k_Matmul_Mish_Mish", "conv", "conv2d instead of a GEMM with the same Mish-Mish epilogue")
a("kb/l2/4_Conv2d_Mish_Mish__bf16", "B", "near", "27k_Matmul_Mish_Mish", "conv", "bf16 variant of kb/l2/4")
# ---------------- Gemini-era pool (gemma-dataset-gen) near-dups ----------------
a("authored/norm_gated/mamba2_gated_rmsnorm_2", "B", "near", "12p_RMSNorm", "norm", "Mamba2 gated RMSNorm: x*silu(z) then RMSNorm (extra gate input)")
a("authored/norm_gated/mamba2_gated_rmsnorm_3", "B", "near", "12p_RMSNorm", "norm", "Mamba2 gated RMSNorm variant")
a("authored/moe_expert_ffn_batched/moe_expert_ffn_batched_2", "B", "far", "14p_Ragged_Dot", "moe", "batched per-expert gated FFN (three GEMMs + activation), not a bare grouped matmul")
a("kb/l1/50_conv_standard_2D__square_input__square_kernel", "B", "near", "18k_Conv2D_ReLU_BiasAdd", "conv", "plain conv2d; 18k adds ReLU and a bias")
a("kb/l2/19_ConvTranspose2d_GELU_GroupNorm", "B", "far", "25k_Conv3d_GroupNorm_Mean", "conv", "different KernelBench L2 problem")
a("kb/l2/27_Conv3d_HardSwish_GroupNorm_Mean", "B", "near", "25k_Conv3d_GroupNorm_Mean", "conv", "25k's chain plus a HardSwish")
a("kb/l2/31_Conv2d_Min_Add_Multiply", "B", "far", "18k_Conv2D_ReLU_BiasAdd", "conv", "different KernelBench L2 problem")
a("kb/l2/33_Gemm_Scale_BatchNorm", "B", "near", "31k_Gemm_BatchNorm_GELU_ReLU", "gemm_epilogue", "GEMM+scale+BatchNorm vs GEMM+BatchNorm+GELU+ReLU")
a("kb/l2/32_Conv2d_Scaling_Min", "B", "far", "39k_Conv2d_GELU_GlobalAvgPool", "conv", "different KernelBench L2 problem")
a("kb/l2/39_Gemm_Scale_BatchNorm", "B", "near", "31k_Gemm_BatchNorm_GELU_ReLU", "gemm_epilogue", "GEMM+scale+BatchNorm vs GEMM+BatchNorm+GELU+ReLU")
a("kb/l2/64_Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU", "B", "far", "42k_Gemm_Max_Subtract_GELU", "gemm_epilogue", "different KernelBench L2 problem")
a("kb/l2/63_Gemm_ReLU_Divide", "A", "", "41k_Gemm_Add_ReLU", "gemm_epilogue", "Linear(+bias) -> ReLU -> /const = 41k (and 19k) modulo a scalar; excluded only in the Gemini-era pool, CURRENTLY IN the Kimi pool")
a("kb/l2/63_Gemm_ReLU_Divide__bf16", "A", "", "41k_Gemm_Add_ReLU", "gemm_epilogue", "bf16 variant of kb/l2/63; never flagged by any filter, CURRENTLY IN the Kimi pool")
a("kb/l2/73_Conv2d_BatchNorm_Scaling", "B", "near", "34k_Conv2d_Activation_BatchNorm", "conv", "conv+BN+scale vs conv+activation+BN")
a("kb/l2/83_Conv3d_GroupNorm_Min_Clamp_Dropout", "B", "near", "25k_Conv3d_GroupNorm_Mean", "conv", "25k's conv3d+GN with min/clamp instead of mean")
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
