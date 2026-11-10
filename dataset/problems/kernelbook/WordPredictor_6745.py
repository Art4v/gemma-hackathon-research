"""WordPredictor: mean -> Linear+ReLU -> concat -> Linear+ReLU attention -> softmax(time) -> weighted sum -> concat -> Linear+ReLU -> Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'WordPredictor_6745',
    'seq_len': 512,
    'batch': 64,
    'encoder_output_dim': 1024,
    'hidden_dim': 1024,
    'output_dim': 1024,
    'attn_output_dim': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    seq_len = CONFIG['seq_len']
    batch = CONFIG['batch']
    enc_dim = CONFIG['encoder_output_dim']
    hidden_dim = CONFIG['hidden_dim']
    output_dim = CONFIG['output_dim']
    attn_dim = CONFIG['attn_output_dim']
    scale = CONFIG['weight_scale']

    keys = jax.random.split(jax.random.key(0), 9)

    encoder_hiddens = jax.random.uniform(keys[0], (seq_len, batch, enc_dim), dtype)
    init_w = jax.random.normal(keys[1], (enc_dim, enc_dim), dtype) * scale
    init_b = jax.random.normal(keys[2], (enc_dim,), dtype) * scale
    attn_w = jax.random.normal(keys[3], (2 * enc_dim, attn_dim), dtype) * scale
    attn_b = jax.random.normal(keys[4], (attn_dim,), dtype) * scale
    hidden_w = jax.random.normal(keys[5], (2 * enc_dim, hidden_dim), dtype) * scale
    hidden_b = jax.random.normal(keys[6], (hidden_dim,), dtype) * scale
    output_w = jax.random.normal(keys[7], (hidden_dim, output_dim), dtype) * scale
    output_b = jax.random.normal(keys[8], (output_dim,), dtype) * scale

    return (
        encoder_hiddens,
        init_w,
        init_b,
        attn_w,
        attn_b,
        hidden_w,
        hidden_b,
        output_w,
        output_b,
    )


def workload(
    encoder_hiddens,
    init_w,
    init_b,
    attn_w,
    attn_b,
    hidden_w,
    hidden_b,
    output_w,
    output_b,
):
    init_state = jax.nn.relu(jnp.mean(encoder_hiddens, axis=0) @ init_w + init_b)

    expanded_init = jnp.broadcast_to(init_state[None, ...], encoder_hiddens.shape)
    attn_input = jnp.concatenate([expanded_init, encoder_hiddens], axis=2)
    attn_scores = jax.nn.relu(attn_input @ attn_w + attn_b)
    attn_scores = jax.nn.softmax(attn_scores, axis=0)

    attned_state = jnp.sum(encoder_hiddens * attn_scores, axis=0)
    pred_input = jnp.concatenate([init_state, attned_state], axis=1)
    pred_hidden = jax.nn.relu(pred_input @ hidden_w + hidden_b)
    return pred_hidden @ output_w + output_b
