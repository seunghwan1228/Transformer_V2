import tensorflow as tf
import tensorflow_datasets as tfds

from nlp_data.get_data import GetData

class PreprocessData:
    def __init__(self, data_name):
        self.data_name = data_name
        self.examples, self.metadata = GetData(self.data_name)
        self.train_examples, self.val_examples = self.examples['train'], self.examples['validation']


    def build_tokenizer(self):
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((en.numpy() for pt, en in self.train_examples), target_vocab_size=2**13)
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((pt.numpy() for pt, en in self.train_examples), target_vocab_size=2**13)
        return tokenizer_en, tokenizer_pt


