"""[bf16 variant] BiaffineAttention: Bilinear(x1, x2, no bias) + Linear(concat(x1, x2)) with bias."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BiaffineAttention_10925',
    'batch': 8,
    'seq_len': 1024,
    'in_features': 512,
    'out_features': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    b, s = CONFIG['batch'], CONFIG['seq_len']
    d, o = CONFIG['in_features'], CONFIG['out_features']
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(0), 5)
    x1 = jax.random.uniform(k1, (b, s, d), dtype)
    x2 = jax.random.uniform(k2, (b, s, d), dtype)
    w_bil = jax.random.normal(k3, (o, d, d), dtype) * 0.02
    w_lin = jax.random.normal(k4, (2 * d, o), dtype) * 0.02
    b_lin = jax.random.normal(k5, (o,), dtype) * 0.02
    return x1, x2, w_bil, w_lin, b_lin


def workload(x1, x2, w_bil, w_lin, b_lin):
    # torch.nn.Bilinear: y_k = x1^T W_k x2, W stored (out, in1, in2)
    bil = jnp.einsum('...i,kij,...j->...k', x1, w_bil, x2)
    lin = jnp.concatenate((x1, x2), axis=-1) @ w_lin + b_lin
    return bil + lin
