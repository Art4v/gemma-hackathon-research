"""[bf16 variant] adaILN: rho-blend of instance norm and layer norm (unbiased var), then per-(B,C) gamma scale and beta shift."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'adaILN_10739',
    'batch': 16,
    'num_features': 256,
    'height': 128,
    'width': 128,
    'eps': 1e-5,
    'rho_init': 0.9,
}


def create_inputs(dtype=jnp.bfloat16):
    b, c = CONFIG['batch'], CONFIG['num_features']
    h, w = CONFIG['height'], CONFIG['width']
    kx, kg, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(kx, (b, c, h, w), dtype)
    gamma = jax.random.uniform(kg, (b, c), dtype)
    beta = jax.random.uniform(kb, (b, c), dtype)
    rho = jnp.full((1, c, 1, 1), CONFIG['rho_init'], dtype)
    return x, gamma, beta, rho


def workload(x, gamma, beta, rho):
    eps = CONFIG['eps']
    # torch.std is unbiased by default, hence ddof=1.
    in_mean = jnp.mean(x, axis=(2, 3), keepdims=True)
    in_var = jnp.var(x, axis=(2, 3), keepdims=True, ddof=1)
    out_in = (x - in_mean) / jnp.sqrt(in_var + eps)
    ln_mean = jnp.mean(x, axis=(1, 2, 3), keepdims=True)
    ln_var = jnp.var(x, axis=(1, 2, 3), keepdims=True, ddof=1)
    out_ln = (x - ln_mean) / jnp.sqrt(ln_var + eps)
    out = rho * out_in + (1 - rho) * out_ln
    # Same as torch unsqueeze(2).unsqueeze(3), for any gamma rank.
    g = jnp.expand_dims(gamma, (2, 3))
    bt = jnp.expand_dims(beta, (2, 3))
    return out * g + bt
