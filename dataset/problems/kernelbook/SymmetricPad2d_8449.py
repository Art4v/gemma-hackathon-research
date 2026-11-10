"""Channel split into four groups -> per-group asymmetric ZeroPad2d -> channel concat."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SymmetricPad2d_8449',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'pad': 1,
    'num_groups': 4,
    # Per-group flags for (left, right, top, bottom), multiplied by 'pad'.
    'pad_flags': ((1, 0, 1, 0), (1, 0, 0, 1), (0, 1, 1, 0), (0, 1, 0, 1)),
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def workload(x):
    pad = CONFIG['pad']
    groups = CONFIG['num_groups']
    sub = x.shape[1] // groups
    outputs = []
    for i, (left, right, top, bottom) in enumerate(CONFIG['pad_flags']):
        start = i * sub
        # The last group takes any remainder channels, as x[:, 3 * sub:] does.
        stop = (i + 1) * sub if i < groups - 1 else x.shape[1]
        outputs.append(
            jnp.pad(
                x[:, start:stop],
                ((0, 0), (0, 0), (top * pad, bottom * pad), (left * pad, right * pad)),
                mode='constant',
                constant_values=0,
            )
        )
    return jnp.concatenate(outputs, axis=1)
