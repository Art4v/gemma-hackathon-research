"""[bf16 variant] Quaternion: channel L2-normalize (eps 1e-5) -> 3x3 rotation-matrix entries -> stack -> reshape (-1, 3, 3)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Quaternion_16824',
    'batch': 32,
    'channels': 4,
    'height': 512,
    'width': 512,
    'eps': 1e-05,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    rvec = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (rvec,)


def workload(rvec):
    theta = jnp.sqrt(CONFIG['eps'] + jnp.sum(rvec ** 2, axis=1))
    r = rvec / theta[:, None]
    r0, r1, r2, r3 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    out = jnp.stack(
        (
            1.0 - 2.0 * r1 ** 2 - 2.0 * r2 ** 2,
            2.0 * (r0 * r1 - r2 * r3),
            2.0 * (r0 * r2 + r1 * r3),
            2.0 * (r0 * r1 + r2 * r3),
            1.0 - 2.0 * r0 ** 2 - 2.0 * r2 ** 2,
            2.0 * (r1 * r2 - r0 * r3),
            2.0 * (r0 * r2 - r1 * r3),
            2.0 * (r0 * r3 + r1 * r2),
            1.0 - 2.0 * r0 ** 2 - 2.0 * r1 ** 2,
        ),
        axis=1,
    )
    # Row-major reshape of (B, 9, H, W), matching torch .view(-1, 3, 3) on a contiguous tensor.
    return out.reshape(-1, 3, 3)
