import tensorflow as tf


def create_padding_mask(sequence):
    mask = tf.cast(tf.math.equal(sequence, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]
