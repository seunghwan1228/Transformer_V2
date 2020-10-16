import tensorflow as tf

from train_model.train_utils import create_padding_mask, create_lookahead_mask


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


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, model_dim, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.model_dim = tf.cast(model_dim, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.model_dim) * tf.math.minimum(arg1, arg2)


class TrainModel:
    def __init__(self, model, model_dim):
        self.model = model
        self.model_dim = model_dim
        self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.lr_schedule = CustomSchedule(self.model_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule,
                                                  beta_1=0.9,
                                                  beta_2=0.98,
                                                  epsilon=1e-9)

        self.train_loss = tf.metrics.Mean(name='Train loss')
        self.train_acc = tf.metrics.SparseCategoricalAccuracy(name='Train Acc')

    def loss_fun_one(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_obj(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    def loss_fun_two(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        mask = tf.cast(mask, tf.float32)
        loss_ = self.loss_obj(real, pred, sample_weight=mask)
        return loss_

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                                  tf.TensorSpec(shape=(None, None), dtype=tf.int64)])
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_mask, dec_attn_one_mask, dec_attn_two_mask = create_mask(inp, tar_inp)

        with tf.GradientTape() as tape:
            prediction = self.model(encoder_input=inp,
                                    encoder_mask=enc_mask,
                                    decoder_input=tar_inp,
                                    decoder_mask_1=dec_attn_one_mask,
                                    decoder_mask_2=dec_attn_two_mask,
                                    training=True)
            loss_value = self.loss_fun_one(tar_real, prediction)

        gradient = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

        self.train_loss.update_state(loss_value)
        self.train_acc.update_state(tar_real, prediction)
