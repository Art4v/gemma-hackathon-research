"""NLgate: flatten(q + softmax(q @ k_view) @ v_view), with k and v both taken from x[1]."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NLgate_13812',
    'num_slices': 2,
    'batch': 16,
    'thw_dim': 1024,
    'c_dim': 1024,
    'tf_dim': 1024,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['num_slices'], CONFIG['batch'], CONFIG['thw_dim'], CONFIG['c_dim'])
    x = jax.random.uniform(key, shape, dtype=dtype)
    return (x,)


def workload(x):
    # ponytail: thw == tf is forced by torch's batched matmul (q and k have equal numel).
    thw = x.shape[-2]
    c = x.shape[-1]
    tf = thw
    qin = x[0].reshape(-1, thw, c)
    kin = x[1].reshape(-1, c, tf)
    vin = x[1].reshape(-1, tf, c)
    scores = jnp.matmul(qin, kin)
    attn = jax.nn.softmax(scores, axis=-1)
    out = qin + jnp.matmul(attn, vin)
    return out.reshape(out.shape[0], -1)
