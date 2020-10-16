import tensorflow as tf


def create_padding_mask(sequence):
    mask = tf.cast(tf.math.equal(sequence, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_lookahead_mask(size):
    return 1 - tf.linalg.band_part(tf.ones(shape=(size, size)), num_lower=-1, num_upper=0)