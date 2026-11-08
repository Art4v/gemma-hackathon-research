"""Per-head gated RMSNorm: rmsnorm(x) * weight * silu(gate), as in Gated DeltaNet / Mamba2 RMSNormGated."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gated_rmsnorm_per_head',
    'batch': 4,
    'seq_len': 4096,
    'num_heads': 32,
    'head_dim': 128,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.float32):
    B = CONFIG['batch']
    T = CONFIG['seq_len']
    H = CONFIG['num_heads']
    D = CONFIG['head_dim']
    k_x, k_g = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k_x, (B, T, H, D), dtype)
    gate = jax.random.normal(k_g, (B, T, H, D), dtype)
    weight = jnp.ones((D,), dtype)
    return (x, gate, weight)


def workload(x, gate, weight):
    eps = CONFIG['eps']
    x32 = x.astype(jnp.float32)
    g32 = gate.astype(jnp.float32)
    w32 = weight.astype(jnp.float32)
    var = jnp.mean(x32 * x32, axis=-1, keepdims=True)
    y = x32 * jax.lax.rsqrt(var + eps)
    y = y * w32
    y = y * (g32 * jax.nn.sigmoid(g32))
    return y.astype(x.dtype)
