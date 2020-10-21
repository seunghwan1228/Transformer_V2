import tensorflow as tf


def create_padding_mask(sequence):
    mask = tf.cast(tf.math.equal(sequence, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_lookahead_mask(size):
    return 1 - tf.linalg.band_part(tf.ones(shape=(size, size)), num_lower=-1, num_upper=0)


def create_mask(inp, tar):
    # Encoder mask
    enc_padding_mask = create_padding_mask(inp)
    # Decoder mask: 1st
    lookahead_mask = create_lookahead_mask(tf.shape(tar)[1])
    dec_one_mask = create_padding_mask(tar)
    dec_complex_mask = tf.math.maximum(dec_one_mask, lookahead_mask)
    # Decoder mask: 2nd
    dec_padding_mask = create_padding_mask(inp)
    return enc_padding_mask, dec_complex_mask, dec_padding_mask