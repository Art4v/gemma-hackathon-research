# Algorithm-level contamination review (rule 3)

Question for each problem: **is its core computation the same algorithm as one of the 64 JAXBench tasks** (files in `jaxbench/`),
differing only in shapes, dtypes, constants, or trivial pre/post-processing (a scale, a bias add, one activation, a reshape)?
Score-based filters (op chains, Jaccard) already ran; this review catches same-algorithm problems that look different as op graphs.

**CORRECTION (read first): attention variants with biases, score mods or masks are the SAME algorithm as JAXBench tasks, not a keep.**
1p/2p/4p are softmax(QK^T)V (causal, MHA/GQA; 4p is splash attention, which handles arbitrary masks), and **5p_Flex_Attention is causal
softmax(QK^T*scale + rel_pos_bias)V, i.e. attention with an arbitrary score modification**. So EXCLUDE (closest 1p/2p/4p/5p) any problem
whose core is softmax(QK^T + mod)V where mod is an additive bias (ALiBi, T5/relative-position, learned or decomposed relative bias,
Transformer-XL/Music-Transformer relative terms), an elementwise score mod (softcap, temperature) or a mask (causal, sliding window,
local/global, padding, block). Projected-K/V attention (Linformer) is SDPA on projected keys: exclude.

**EXCLUDE** when the problem's core (what dominates its FLOPs/bytes and its kernel design) is a JAXBench task's algorithm. Examples:
- plain scaled-dot-product attention softmax(QK^T)V, causal or not, MHA/GQA/MQA -> 1p/2p (and MLA -> 3p/53p/54p, block/sparse masks -> 4p, flex-style score mods -> 5p, paged/ragged KV -> 6p/7p/60p/61p, quantized attention -> 65p)
- standard softmax cross-entropy / NLL over logits with integer targets -> 13p; a fused final-linear + softmax-CE -> 64p
- RMSNorm (x * rsqrt(mean(x^2)+eps) * w) -> 12p
- a plain linear/matmul with at most a bias add -> 8p; SwiGLU/GeGLU gated MLP (up*act(gate) then down) -> 9p
- top-k gated mixture of experts -> 10p; grouped/ragged matmul over token groups -> 11p/14p/59p
- retention -> 15p; Mamba2 SSD / selective scan -> 16p; gated delta net -> 63p; causal depthwise conv1d + gate -> 62p
- triangle multiplicative update -> 17p; Sinkhorn normalisation -> 56p (and the other DeepSeek-V4 tasks 51p-58p)
- the exact op chain of a KernelBench-style k-task (e.g. Gemm -> Sigmoid -> LogSumExp) -> that k-task

**KEEP** when the math is materially different, or the shared algorithm is a minor part of a larger computation. Examples:
- attention whose core is NOT softmax(QK^T + modification)V: linear/kernelized attention without softmax, stick-breaking, AFT,
  synthesizer with fixed/random attention matrices, additive MLP-score (Bahdanau) attention, softmax taken over separate axes
  ("efficient attention"). Luong-general (a learned matrix between Q and K, then softmax and V) is borderline: judge the core.
- channel/spatial "attention" gates (squeeze-excitation, CBAM): sigmoid gates, not softmax(QK^T)V
- focal loss, dice loss, label-smoothed or KL losses (different math from plain CE; label smoothing alone is borderline)
- LayerNorm / GroupNorm / InstanceNorm (centred: not RMSNorm), convolutions, pooling, plain elementwise ops
- a ResNet/transformer block where attention or an MLP is one part among several substantial parts

**BORDERLINE** when you can't decide; say why. The team excludes borderline ones by default, so use it honestly, not as a hedge.
