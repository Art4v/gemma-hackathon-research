"""[bf16 variant] Softmax(dim=1) -> max over dim=1 -> threshold compare (cond_up, cond_down)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LogitCond_8451',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 256,
    'thres': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(outputs):
    probs = jax.nn.softmax(outputs, axis=1)
    max_probs = jnp.max(probs, axis=1)
    thres = CONFIG['thres']
    cond_up = max_probs > thres
    cond_down = max_probs <= thres
    return cond_up, cond_down
