"""[bf16 variant] Cartesian: polar (r, phi) in last axis -> stack(r*cos(phi), r*sin(phi))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Cartesian_13697',
    'batch': 16,
    'dim1': 512,
    'dim2': 1024,
    'channels': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['channels']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    r = x[..., 0]
    phi = x[..., 1]
    return jnp.stack((r * jnp.cos(phi), r * jnp.sin(phi)), axis=-1)
