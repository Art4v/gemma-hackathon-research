"""[bf16 variant] BPR loss: double-cast difference -> sigmoid -> log -> negated mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BPR_17179',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'dim3': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['dim3'])
    key_pos, key_neg = jax.random.split(jax.random.key(0))
    pos = jax.random.uniform(key_pos, shape, dtype)
    neg = jax.random.uniform(key_neg, shape, dtype)
    return pos, neg


def workload(pos, neg):
    # Default float type: f64 under jax_enable_x64, otherwise f32.
    wide = jnp.result_type(float)
    loss = jnp.log(jax.nn.sigmoid(pos.astype(wide) - neg.astype(wide)))
    # Axis-by-axis mean equals the global mean and avoids f32 drift in one huge sum.
    for _ in range(loss.ndim):
        loss = jnp.mean(loss, axis=-1)
    return -loss
