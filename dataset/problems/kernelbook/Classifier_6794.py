"""avg_pool2d(8) -> flatten -> Linear(900, 3) -> log_softmax(dim=1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Classifier_6794',
    'batch': 1024,
    'channels': 4,
    'height': 121,
    'width': 121,
    'pool': 8,
    'in_features': 900,
    'out_features': 3,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    w = jax.random.normal(kw, (CONFIG['in_features'], CONFIG['out_features']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['out_features'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    p = CONFIG['pool']
    n, c, h, wd = x.shape
    ho, wo = h // p, wd // p
    # Floor mode: trailing rows/cols that do not fill a window are dropped, as in PyTorch.
    xc = x[:, :, :ho * p, :wo * p]
    pooled = xc.reshape(n, c, ho, p, wo, p).mean(axis=(3, 5))
    feats = pooled.reshape(-1, w.shape[0])
    logits = feats @ w + b
    return jax.nn.log_softmax(logits, axis=1)
