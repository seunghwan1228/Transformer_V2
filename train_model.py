import tensorflow as tf
import tensorflow_datasets as tfds

from config_model.model_config import ModelConfig
from model.model_transformer import Transformer
from nlp_data.data_prep import PreprocessData
from model_trainer.train_step import TrainModel



class TrainTransformer:
    def __init__(self, data_name, filter_length, buffer_size, batch_size):
        self.data_name = data_name
        self.filter_length = filter_length
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.config = ModelConfig()
        self.data = PreprocessData(data_name=self.data_name,
                                   filter_length=self.filter_length,
                                   buffer_size=self.buffer_size,
                                   batch_size=self.batch_size)

    def load_data(self):
        train_data, valid_data = self.data()
        return train_data, valid_data


    def build_model(self):
        return Transformer(encoder_vocab_size=self.data.tokenizer_lang_1.vocab_size + 2,
                           decoder_vocab_size=self.data.tokenizer_lang_2.vocab_size + 2,
                           encoder_position_encoding_size=self.data.tokenizer_lang_1.vocab_size + 2,
                           decoder_position_encoding_size=self.data.tokenizer_lang_2.vocab_size + 2,
                           num_heads=self.config.num_heads,
                           model_dim=self.config.model_dim,
                           ffn_units=self.config.ffn_units,
                           num_layers=self.config.num_layers,
                           rate=self.config.dropout_rate)
