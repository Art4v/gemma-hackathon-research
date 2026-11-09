"""[bf16 variant] Biaffine: append ones -> bias-free linear -> reshape -> batched matmul with second input -> transpose/reshape/squeeze."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Biaffine_5538',
    'batch': 16,
    'len1': 1024,
    'len2': 1024,
    'in_features': 1024,
    'out_features': 1,
    'bias1': True,
    'bias2': True,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    b, l1, l2, d = CONFIG['batch'], CONFIG['len1'], CONFIG['len2'], CONFIG['in_features']
    d_in = d + int(CONFIG['bias1'])
    d_out = CONFIG['out_features'] * (d + int(CONFIG['bias2']))
    input1 = jax.random.uniform(k1, (b, l1, d), dtype)
    input2 = jax.random.uniform(k2, (b, l2, d), dtype)
    w = jax.random.normal(k3, (d_in, d_out), dtype) * 0.02
    return input1, input2, w


def _append_ones(x):
    return jnp.concatenate([x, jnp.ones(x.shape[:-1] + (1,), x.dtype)], axis=-1)


def workload(input1, input2, w):
    b, l1, _ = input1.shape
    _, l2, _ = input2.shape
    if CONFIG['bias1']:
        input1 = _append_ones(input1)
    if CONFIG['bias2']:
        input2 = _append_ones(input2)
    d2 = input2.shape[-1]
    out_features = w.shape[1] // d2
    affine = (input1 @ w).reshape(b, l1 * out_features, d2)
    # bmm(affine, input2^T).transpose(1, 2) -> (b, l2, l1*out)
    biaffine = jnp.einsum('bik,bjk->bji', affine, input2)
    biaffine = biaffine.reshape(b, l2, l1, out_features)
    if biaffine.shape[-1] == 1:
        biaffine = jnp.squeeze(biaffine, axis=-1)
    return biaffine
