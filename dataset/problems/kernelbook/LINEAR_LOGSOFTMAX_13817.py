"""[bf16 variant] Linear (last axis) -> LogSoftmax over dim=1 of a 4D tensor."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LINEAR_LOGSOFTMAX_13817',
    'batch': 8,
    'd1': 128,
    'd2': 128,
    'input_dim': 512,
    'nclass': 512,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['d1'], CONFIG['d2'], CONFIG['input_dim']), dtype)
    w = jax.random.normal(kw, (CONFIG['input_dim'], CONFIG['nclass']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['nclass'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    # PyTorch LogSoftmax(dim=1) on a 4D tensor normalizes over axis 1, not the class axis.
    return jax.nn.log_softmax(x @ w + b, axis=CONFIG['softmax_axis'])
