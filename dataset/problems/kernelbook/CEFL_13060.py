"""CEFL: softmax(dim=-1) gather + log_softmax(dim=1) gather -> focal-style weighted CE -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CEFL_13060',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 256,
    'gamma': 1,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_t = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(k_x, shape, dtype)
    target = jax.random.randint(k_t, (CONFIG['batch'],), 0, CONFIG['channels'], dtype=jnp.int32)
    return x, target


def workload(x, target):
    rows = jnp.arange(target.shape[0])
    # torch: prob[range(N), target] indexes dims 0 and 1 -> (N, H, W)
    prob = jax.nn.softmax(x, axis=-1)[rows, target]
    attn = (1.0 - prob) ** CONFIG['gamma']
    ce = -jax.nn.log_softmax(x, axis=1)[rows, target]
    loss = (1.0 - prob) * ce + prob * attn * ce
    return jnp.mean(loss)
