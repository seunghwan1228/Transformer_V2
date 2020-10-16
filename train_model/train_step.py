import tensorflow as tf
from model.model_transformer import Transformer




@tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                              tf.TensorSpec(shape=(None, None), dtype=tf.int64)])
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]


