"""AvgPool2d per timestep (SeqToANNContainer) -> LIF spiking neuron over the time axis."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'APLayer_17312',
    'batch': 8,
    'time_steps': 8,
    'channels': 64,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'tau': 0.5,
    'thresh': 1.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['time_steps'], CONFIG['channels'],
         CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def _avg_pool_last2(x, k):
    # Pools the last two axes for any rank, matching AvgPool2d on the
    # flattened (B*T, ...) tensor whether it is 3-D or 4-D.
    nd = x.ndim
    window = (1,) * (nd - 2) + (k, k)
    s = lax.reduce_window(x, jnp.array(0, x.dtype), lax.add, window, window, 'VALID')
    return s / jnp.array(k * k, x.dtype)


def workload(x):
    k = CONFIG['kernel_size']
    tau = CONFIG['tau']
    thresh = CONFIG['thresh']
    y = _avg_pool_last2(x, k)
    y_t = jnp.moveaxis(y, 1, 0)

    def step(mem, xt):
        mem = mem * tau + xt
        spike = (mem - thresh > 0).astype(xt.dtype)
        mem = (1 - spike) * mem
        return mem, spike

    _, spikes = lax.scan(step, jnp.zeros_like(y_t[0]), y_t)
    return jnp.moveaxis(spikes, 0, 1)
