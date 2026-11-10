"""SpatialGC: 1x1 Conv2d (C_in -> K*C_out) then einsum('nkctv,kvw->nctw') with adjacency A."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SpatialGC_5230',
    'batch': 16,
    'in_channels': 128,
    'out_channels': 128,
    'kernel_size': 4,
    'time': 256,
    'nodes': 128,
}


def create_inputs(dtype=jnp.float32):
    n, cin, cout = CONFIG['batch'], CONFIG['in_channels'], CONFIG['out_channels']
    k, t, v = CONFIG['kernel_size'], CONFIG['time'], CONFIG['nodes']
    kx, ka, kw, kb = jax.random.split(jax.random.key(0), 4)
    x = jax.random.uniform(kx, (n, cin, t, v), dtype)
    a = jax.random.uniform(ka, (k, v, v), dtype)
    w = jax.random.normal(kw, (cin, k * cout), dtype) * 0.02
    b = jax.random.normal(kb, (k * cout,), dtype) * 0.02
    return x, a, w, b


def workload(x, A, w, b):
    k = A.shape[0]
    cin, kc = w.shape
    # 1x1 conv == per-pixel channel matmul; out channel index = k_idx * C_out + c.
    w3 = w.reshape(cin, k, kc // k)
    b3 = b.reshape(k, kc // k)
    y = jnp.einsum('nitv,ikc->nkctv', x, w3, precision=lax.Precision.HIGHEST)
    y = y + b3[None, :, :, None, None]
    out = jnp.einsum('nkctv,kvw->nctw', y, A, precision=lax.Precision.HIGHEST)
    return out, A
