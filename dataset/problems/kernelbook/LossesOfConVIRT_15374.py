"""[bf16 variant] ConVIRT bidirectional contrastive loss: cosine-similarity matrix, row/column log-softmax diagonal, weighted mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LossesOfConVIRT_15374',
    'batch': 8192,
    'dim': 2048,
    'tau': 0.1,
    'lambd': 0.75,
    'eps': 1e-8,
}


def create_inputs(dtype=jnp.bfloat16):
    kv, ku = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['dim'])
    v = jax.random.uniform(kv, shape, dtype)
    u = jax.random.uniform(ku, shape, dtype)
    return v, u


def workload(v, u):
    tau = CONFIG['tau']
    lambd = CONFIG['lambd']
    eps = CONFIG['eps']
    v = v.astype(jnp.float32)
    u = u.astype(jnp.float32)
    # PyTorch cosine_similarity: each vector divided by max(||x||, eps).
    vn = v / jnp.maximum(jnp.linalg.norm(v, axis=1, keepdims=True), eps)
    un = u / jnp.maximum(jnp.linalg.norm(u, axis=1, keepdims=True), eps)
    logits = jnp.matmul(vn, un.T, precision=jax.lax.Precision.HIGHEST) / tau
    diag = jnp.diagonal(logits)
    # image_text(i): softmax over row i; text_image(i): softmax over column i.
    loss_it = jax.nn.logsumexp(logits, axis=1) - diag
    loss_ti = jax.nn.logsumexp(logits, axis=0) - diag
    return jnp.mean(lambd * loss_it + (1.0 - lambd) * loss_ti)
