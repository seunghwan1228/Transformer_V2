import tensorflow as tf
import tensorflow_datasets as tfds

from nlp_data.get_data import GetData


class PreprocessData:
    def __init__(self, data_name, filter_length, buffer_size, batch_size):
        self.data_name = data_name
        self.filter_length = filter_length
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.examples, self.metadata = GetData(self.data_name).get_data()
        self.train_examples, self.val_examples = self.examples['train'], self.examples['validation']

    def build_tokenizer(self):
        tokenizer_lang_1 = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in self.train_examples), target_vocab_size=2 ** 13)
        tokenizer_lang_2 = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in self.train_examples), target_vocab_size=2 ** 13)

        self.tokenizer_lang_1 = tokenizer_lang_1
        self.tokenizer_lang_2 = tokenizer_lang_2

        return tokenizer_lang_1, tokenizer_lang_2

    def get_data_sample(self):
        lang_one_sample = []
        lang_two_sample = []

        sample_data = next(iter(self.train_examples))
        lang_one_sample.append(sample_data[0].numpy())
        lang_two_sample.append(sample_data[1].numpy())

        print(f'Language One: {lang_one_sample}\n')
        print(f'Language Two: {lang_two_sample}\n')

    def encode_text(self, lang1, lang2):
        lang1 = [self.tokenizer_lang_1.vocab_size] + self.tokenizer_lang_1.encode(lang1.numpy()) + [
            self.tokenizer_lang_1.vocab_size + 1]
        lang2 = [self.tokenizer_lang_2.vocab_size] + self.tokenizer_lang_2.encode(lang2.numpy()) + [
            self.tokenizer_lang_2.vocab_size + 1]
        return lang1, lang2

    def tf_encode_text(self, lang1, lang2):
        lang1_result, lang2_result = tf.py_function(self.encode_text, [lang1, lang2], [tf.int64, tf.int64])
        lang1_result.set_shape([None])
        lang2_result.set_shape([None])
        return lang1_result, lang2_result

    def filter_data(self, x, y):
        return tf.logical_and(tf.size(x) <= self.filter_length, tf.size(y) <= self.filter_length)

    def process_data(self, dataset):
        dataset = dataset.map(self.tf_encode_text)
        dataset = dataset.filter(self.filter_data)
        dataset = dataset.cache()
        dataset = dataset.shuffle(self.buffer_size)
        dataset = dataset.padded_batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def __call__(self):
        print('Data Sample\n')
        self.get_data_sample()

        print('Building Tokenizer\n')
        self.build_tokenizer()

        train_data = self.process_data(self.train_examples)
        valid_data = self.process_data(self.val_examples)
        return train_data, valid_data

    def tpu_process_data(self, dataset, batch_size):
        dataset = dataset.map(self.tf_encode_text)
        dataset = dataset.filter(self.filter_data)
        dataset = dataset.cache()
        dataset = dataset.shuffle(self.buffer_size)
        dataset = dataset.padded_batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def prepare_tpu_data(self, batch_size):
        train_data = self.tpu_process_data(self.train_examples, batch_size)
        valid_data = self.tpu_process_data(self.val_examples, batch_size)
        return train_data, valid_data



if __name__ == '__main__':
    # Initialize object
    pdata = PreprocessData('ted_hrlr_translate/pt_to_en', 100, 20000, 64)

    # Build Tokenizer
    tokenizer_pt, tokenizer_en = pdata.build_tokenizer()

    # Test Tokenizer
    sample_string = 'Transformer is awesome.'
    tokenized_string = tokenizer_en.encode(sample_string)
    for ts in tokenized_string:
        print('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))

    # Show lang1 & lang2 samples
    pdata.get_data_sample()

    # Specify data
    train_data, valid_data = pdata.train_examples, pdata.val_examples
    # Convert TF-Dataset
    train_data_converted = train_data.map(pdata.tf_encode_text)

    # Check Data
    next(iter(train_data_converted))

    # TF-Data Created
    train_data, val_data = pdata()

    # Show encoded data
    print(next(iter(train_data)))
