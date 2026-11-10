"""LayerNormAVG: global sqrt(desired_avg * features / sum(x^2)) scale times unbiased-std LayerNorm over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LayerNormAVG_2911',
    'batch': 8,
    'dim1': 32,
    'dim2': 256,
    'features': 1024,
    'desired_avg': 4.0,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['features'])
    x = jax.random.uniform(key, shape, dtype=dtype)
    return (x,)


def workload(x):
    # features == last dim in the KernelBench init; read it from the shape so small test inputs stay consistent.
    features = x.shape[-1]
    to_norm = jnp.sqrt(CONFIG['desired_avg'] * features / jnp.sum(x ** 2))
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True, ddof=1)  # torch.std is unbiased by default
    return to_norm * ((x - mean) / (std + CONFIG['eps']))
