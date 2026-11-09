"""[bf16 variant] KLLoss: log_softmax(pred, dim=1) vs softmax(label*10, dim=1), KLDivLoss mean reduction, times batch size."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'KLLoss_14745',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'label_scale': 10.0,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    prediction = jax.random.uniform(k1, shape, dtype)
    label = jax.random.uniform(k2, shape, dtype)
    return prediction, label


def workload(prediction, label):
    axis = CONFIG['softmax_axis']
    batch_size = prediction.shape[0]
    log_p = jax.nn.log_softmax(prediction, axis=axis)
    log_t = jax.nn.log_softmax(label * CONFIG['label_scale'], axis=axis)
    t = jnp.exp(log_t)
    # KLDivLoss(size_average=True) means reduction='mean': average over every element.
    return jnp.mean(t * (log_t - log_p)) * batch_size
