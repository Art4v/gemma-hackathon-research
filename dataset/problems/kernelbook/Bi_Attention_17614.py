"""[bf16 variant] Bi_Attention: bmm(x1, x2^T) -> masked softmax over dim2 / dim1 -> bmm back onto x2 / x1."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Bi_Attention_17614',
    'batch': 16,
    'x1_len': 1024,
    'x2_len': 2048,
    'dim': 512,
    'mask_threshold': 0.5,
    'inf': 1e13,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    b, l1, l2, d = CONFIG['batch'], CONFIG['x1_len'], CONFIG['x2_len'], CONFIG['dim']
    in_x1 = jax.random.uniform(k1, (b, l1, d), dtype)
    in_x2 = jax.random.uniform(k2, (b, l2, d), dtype)
    x1_len = jax.random.uniform(k3, (b, l1, 1), dtype)
    x2_len = jax.random.uniform(k4, (b, l2, 1), dtype)
    return in_x1, in_x2, x1_len, x2_len


def workload(in_x1, in_x2, x1_len, x2_len):
    b, l1, _ = in_x1.shape
    l2 = in_x2.shape[1]
    hp = lax.Precision.HIGHEST
    att = jnp.einsum('bid,bjd->bij', in_x1, in_x2, precision=hp)
    neg = -CONFIG['inf']
    thr = CONFIG['mask_threshold']
    a_mask = (x1_len <= thr).astype(att.dtype) * neg
    a_mask = jnp.broadcast_to(jnp.reshape(a_mask, (b, l1, -1)), (b, l1, l2))
    b_mask = (x2_len <= thr).astype(att.dtype) * neg
    b_mask = jnp.broadcast_to(jnp.reshape(b_mask, (b, -1, l2)), (b, l1, l2))
    attention_a = jax.nn.softmax(att + a_mask, axis=2)
    attention_b = jax.nn.softmax(att + b_mask, axis=1)
    out_x1 = jnp.einsum('bij,bjd->bid', attention_a, in_x2, precision=hp)
    out_x2 = jnp.einsum('bij,bid->bjd', attention_b, in_x1, precision=hp)
    return out_x1, out_x2
