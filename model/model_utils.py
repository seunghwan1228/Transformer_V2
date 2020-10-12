import tensorflow as tf
import numpy as np


# pos embedding
def get_angle(pos, i, model_dim):
    angle_rates = 1. / np.power(10000, (2* (i//2))/np.float32(model_dim))
    return pos * angle_rates


def positional_encoding(position, model_dim):
    angle_rads = get_angle(np.arange(position)[:, np.newaxis],  # pos, 1
                           np.arange(model_dim)[np.newaxis, :], # 1, model_dim
                           model_dim)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask):
    qk = tf.matmul(q, k, transpose_b=True)
    scaler_dim_k = tf.cast(tf.shape(k)[-1], tf.float32)

    scaled_logits = qk / tf.math.sqrt(scaler_dim_k)
    if mask is not None:
        scaled_logits += (mask * -1e9)
    attn_weight = tf.nn.softmax(scaled_logits, axis=-1)
    context = tf.matmul(attn_weight, v)

    return context, attn_weight