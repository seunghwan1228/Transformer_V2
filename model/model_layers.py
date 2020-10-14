import tensorflow as tf
from model.model_utils import scaled_dot_product_attention, positional_encoding


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, model_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.head_depth = self.model_dim // self.num_heads
        self.wq = tf.keras.layers.Dense(units=model_dim)
        self.wk = tf.keras.layers.Dense(units=model_dim)
        self.wv = tf.keras.layers.Dense(units=model_dim)
        self.dense = tf.keras.layers.Dense(units=model_dim)

    def split_head(self, x):
        split_batch = tf.shape(x)[0]
        x = tf.reshape(x, shape=(split_batch, -1, self.num_heads, self.head_depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def concat_head(self, x):
        concat_batch = tf.shape(x)[0]
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, shape=(concat_batch, -1, self.model_dim))

    def call(self, q, k, v, mask):
        z_q = self.wq(q)
        z_k = self.wk(k)
        z_v = self.wv(v)
        z_h_q = self.split_head(z_q)
        z_h_k = self.split_head(z_k)
        z_h_v = self.split_head(z_v)
        context, attn_weight = scaled_dot_product_attention(z_h_q, z_h_k, z_h_v, mask)
        context = self.concat_head(context)
        mha_out = self.dense(context)

        return mha_out, attn_weight


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, ffn_units, model_dim):
        super(FeedForward, self).__init__()
        self.ffn_units = ffn_units
        self.model_dim = model_dim

        self.ffn_1 = tf.keras.layers.Dense(ffn_units, activation='relu')
        self.ffn_2 = tf.keras.layers.Dense(model_dim)

    def call(self, inputs):
        x = self.ffn_1(inputs)
        return self.ffn_2(x)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, model_dim, ffn_units, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.ffn_units = ffn_units
        self.rate = rate

        self.mha = MultiHeadAttention(self.num_heads, self.model_dim)
        self.mha_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha_dr = tf.keras.layers.Dropout(self.rate)
        self.ffn = FeedForward(self.ffn_units, self.model_dim)
        self.ffn_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn_dr = tf.keras.layers.Dropout(self.rate)

    def call(self, q, k, v, mask, training):
        input_res = q
        context, attn_weight = self.mha(q, k, v, mask)
        context = self.mha_dr(context, training=training)
        context_1 = self.ln(context + input_res)

        ffn_out = self.ffn(context_1)
        ffn_out = self.ffn_dr(ffn_out, training=training)
        ffn_out_1 = self.ffn_ln(ffn_out + context_1)

        return ffn_out_1


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, model_dim, ffn_units, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.ffn_units = ffn_units
        self.rate = rate

        self.mha_1 = MultiHeadAttention(self.num_heads, self.model_dim)
        self.mha_1_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha_1_dr = tf.keras.layers.Dropout(rate)

        self.mha_2 = MultiHeadAttention(self.num_heads, self.model_dim)
        self.mha_2_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha_2_dr = tf.keras.layers.Dropout(rate)

        self.ffn = FeedForward(self.ffn_units, self.model_dim)
        self.ffn_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn_dr = tf.keras.layers.Dropout(rate)

    def call(self, decoder_input, encoder_output, mha_1_mask, mha_2_mask, training):
        input_res = decoder_input
        attn1, attn1_weight = self.mha_1(decoder_input, decoder_input, decoder_input, mha_1_mask)
        attn1 = self.mha_1_dr(attn1)
        out1 = self.mha_1_ln(attn1 + input_res)

        attn2, attn2_weight = self.mha_2(q=out1, k=encoder_output, v=encoder_output, mask=mha_2_mask)
        attn2 = self.mha_2_dr(attn2)
        out2 = self.mha_ln(attn2 + out1)

        ffn_out = self.ffn(out2)
        ffn_out = self.ffn_dr(ffn_out)
        ffn_out = self.ffn_lr(ffn_out + out2)

        return ffn_out, attn1_weight, attn2_weight



class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_heads, model_dim, ffn_units, num_layers, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.ffn_units = ffn_units
        self.num_layers = num_layers
        self.input_vocab_size = input_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate

        self.embedding = tf.keras.layers.Embedding(self.input_vocab_size, self.model_dim)
        self.positional_encoding = positional_encoding(self.maximum_position_encoding, self.model_dim)
        self.embedding_dr =  tf.keras.layers.Dropout(self.rate)

        self.enc_layers = [EncoderLayer(num_heads=self.num_heads, model_dim=self.model_dim, ffn_units=self.ffn_units, rate=self.rate) for _ in range(self.num_layers)]


    def call(self, encoder_input, mask, training):
        seq_len = tf.shape(encoder_input)[1]
        x = self.embedding(encoder_input)
        x *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        x += self.positional_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](q=x, k=x, v=x, mask=mask, training=training)

        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_heads, model_dim, ffn_units, num_layers, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.ffn_units = ffn_units
        self.num_layers = num_layers
        self.input_vocab_size = input_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate

        self.embedding = tf.keras.layers.Embedding(self.input_vocab_size, self.model_dim)
        self.position_encoding = positional_encoding(self.maximum_position_encoding, self.model_dim)
        self.embedding_dr = tf.keras.layers.Dropout(self.rate)

        self.dec_layers = [DecoderLayer(num_heads=self.num_heads, model_dim=self.model_dim, ffn_units=self.ffn_units, rate=self.rate) for _ in range(self.num_layers)]

    def call(self, decoder_input, encoder_output, mask_1, mask_2, training):
        seq_len = tf.shape(decoder_input)[1]
        x = self.embedding(decoder_input)
        x *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        x += self.position_encoding[:, :seq_len, :]
        x = self.embedding_dr(x, training=training)

        for i in range(self.num_layers):
            x, attn1, attn2 = self.dec_layers[i](x, encoder_output, mask_1, mask_2, training)

        return x
