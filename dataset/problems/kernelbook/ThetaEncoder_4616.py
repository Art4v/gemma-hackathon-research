"""[bf16 variant] ThetaEncoder: concat(theta, t0+t1, t0-t1) -> stack[x, sin(x*2^i*w), cos(x*2^i*w)] -> reshape(B, L, -1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ThetaEncoder_4616',
    'batch': 16,
    'lead_num': 128,
    'mid': 2048,
    'last': 4,
    'encoder_len': 4,
    'omega': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    theta = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['lead_num'], CONFIG['mid'], CONFIG['last']),
        dtype=dtype,
    )
    return (theta,)


def workload(theta):
    b, lead_num = theta.shape[0], theta.shape[1]
    t0 = theta[..., 0:1]
    t1 = theta[..., 1:2]
    before = jnp.concatenate([theta, t0 + t1, t0 - t1], axis=-1)
    outs = [before]
    for i in range(CONFIG['encoder_len']):
        arg = before * (2 ** i) * CONFIG['omega']
        outs += [jnp.sin(arg), jnp.cos(arg)]
    return jnp.stack(outs, axis=-1).reshape(b, lead_num, -1)
