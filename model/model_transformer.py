import tensorflow as tf
from model.model_layers import Encoder, Decoder


class Transformer(tf.keras.Model):
    def __init__(self,
                 encoder_vocab_size,
                 decoder_vocab_size,
                 encoder_position_encoding_size,
                 decoder_position_encoding_size,
                 num_heads,
                 model_dim,
                 ffn_units,
                 num_layers,
                 rate=0.1):
        super(Transformer, self).__init__()

        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.encoder_position_encoding_size = encoder_position_encoding_size
        self.decoder_position_encoding_size = decoder_position_encoding_size
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.ffn_units = ffn_units
        self.num_layers = num_layers
        self.rate = rate

        self.encoder = Encoder(num_heads = self.num_heads,
                               modeldim = self.model_dim,
                               ffn_units=self.ffn_units,
                               num_layers = self.num_layers,
                               input_vocab_size=self.encoder_vocab_size,
                               maximum_position_encoding=self.encoder_position_encoding_size,
                               rate=self.rate)

        self.decoder = Decoder(num_heads=self.num_heads,
                               model_dim=self.model_dim,
                               ffn_units=self.ffn_units,
                               num_layers=self.num_layers,
                               input_vocab_size=self.decoder_vocab_size,
                               maximum_position_encoding=self.decoder_position_encoding_size,
                               rate=self.rate)

        # The activation will be the softmax, but the model output is only logits
        self.linear_out = tf.keras.layers.Dense(units=self.decoder_vocab_size)

    def call(self, encoder_input, encoder_mask, decoder_input, decoder_mask_1, decoder_mask_2, training):

        enc_output = self.encoder(encoder_input, encoder_mask, training)
        decoder_output = self.decoder(decoder_input, enc_output, decoder_mask_1, decoder_mask_2, training)
        final_out = self.linear_out(decoder_output)
        return final_out