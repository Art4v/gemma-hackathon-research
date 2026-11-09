"""[bf16 variant] FlowfieldDiscountedLoss: MSE(none) * prod(flowfield**2 <= 1 mask, axis=-1) broadcast over channels -> sum."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FlowfieldDiscountedLoss_8036',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
    'flow_dim': 2,
    'flow_range': 1.2,
    'threshold': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    b, c, h, w = CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']
    r = CONFIG['flow_range']
    source = jax.random.uniform(k1, (b, c, h, w), dtype)
    target = jax.random.uniform(k2, (b, c, h, w), dtype)
    flowfield = jax.random.uniform(k3, (b, h, w, CONFIG['flow_dim']), dtype, minval=-r, maxval=r)
    return source, target, flowfield


def workload(source, target, flowfield):
    loss = (source - target) ** 2
    sq = flowfield ** 2
    t = CONFIG['threshold']
    # Mirrors torch's two in-place writes: <=t -> 1, >t -> 0, NaN passes through.
    bad_bits = jnp.where(sq > t, jnp.zeros_like(sq), jnp.where(sq <= t, jnp.ones_like(sq), sq))
    mask = jnp.prod(bad_bits, axis=3)[:, None, :, :]
    return jnp.sum(loss * mask)
