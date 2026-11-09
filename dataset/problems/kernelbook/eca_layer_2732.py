"""ECA layer: global avg pool -> 1D conv across channels -> sigmoid -> channel-wise scale."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'eca_layer_2732',
    'batch': 16,
    'channels': 512,
    'height': 64,
    'width': 64,
    'k_size': 3,
}


def create_inputs(dtype=jnp.float32):
    kx, kw = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    w = jax.random.normal(kw, (CONFIG['k_size'],), dtype) * 0.02
    return x, w


def workload(x, w):
    k = w.shape[0]
    c = x.shape[1]
    pad = (k - 1) // 2
    y = jnp.mean(x, axis=(2, 3))  # (B, C)
    yp = jnp.pad(y, ((0, 0), (pad, pad)))
    # Cross-correlation along channels, matching nn.Conv1d (no kernel flip).
    z = sum(w[j] * yp[:, j:j + c] for j in range(k))
    s = jax.nn.sigmoid(z)
    return x * s[:, :, None, None]
