import tensorflow as tf
import tensorflow_datasets as tfds

from config_model.model_config import ModelConfig
from model.model_transformer import Transformer
from nlp_data.data_prep import PreprocessData
from model_trainer.train_step import TrainModel, CustomSchedule
from model_trainer.train_utils import create_mask



class TrainTransformer:
    def __init__(self, data_name, filter_length, buffer_size, batch_size, use_tpu=False):
        self.data_name = data_name
        self.filter_length = filter_length
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.use_tpu = use_tpu
        self.config = ModelConfig()
        self.data = PreprocessData(data_name=self.data_name,
                                   filter_length=self.filter_length,
                                   buffer_size=self.buffer_size,
                                   batch_size=self.batch_size)

        if self.use_tpu:
            self.tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
            tf.config.experimental_connect_to_cluster(self.tpu)
            tf.tpu.experimental.initialize_tpu_system(self.tpu)
            self.tpu_strategy = tf.distribute.experimental.TPUStrategy(self.tpu)

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

    def train_model(self):
        train_data, valid_data = self.load_data()
        model = self.build_model()
        trainer = TrainModel(model=model,
                             model_dim=self.config.model_dim)
        # Train model
        trainer.train_model(train_data)


    def train_model_tpu(self):
        if self.tpu:
            print('TPU Cluster Initialize\n')
            print('Running on TPU ', self.tpu.cluster_spec().as_dict()['worker'])

            per_replica_batch_size = self.batch_size // self.tpu_strategy.num_replicas_in_sync
            train_data, valid_data = self.data.prepare_tpu_data(per_replica_batch_size)

            # TODO Requires to build tf.function
            # Reference : https://www.tensorflow.org/guide/tpu

            with self.tpu_strategy.scope():
                model = self.build_model()
                optimizer = tf.keras.optimizers.Adam(learning_rate=CustomSchedule(self.config.model_dim),
                                                     beta_1=0.9,
                                                     beta_2=0.98,
                                                     epsilon=1e-9)
                training_mean = tf.keras.metrics.Mean(name='Train loss')
                training_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='Train Acc')

            @tf.function
            def train_step(self, iterator):
                def step_fn(inp, tar):
                    tar_inp = tar[:, :-1]
                    tar_real = tar[:, 1:]
                    enc_mask, dec_attn_one_mask, dec_attn_two_mask = create_mask(inp, tar_inp)
                    with tf.GradientTape() as tape:
                        prediction = model(encoder_input=inp,
                                           encoder_mask=enc_mask,
                                           decoder_input=tar_inp,
                                           decoder_mask_1=dec_attn_one_mask,
                                           decoder_mask_2=dec_attn_two_mask,
                                           training=True)
                        loss_value = self.loss_fun_one(tar_real, prediction)
                        loss_value = tf.nn.compute_average_loss(loss_value, global_batch_size=self.batch_size)

                    gradient = tape.gradient(loss_value, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

                    training_mean.update_state(loss_value * self.tpu_strategy.num_replicas_in_sync)
                    training_acc.update_state(tar_real, prediction)

                self.tpu_strategy.run(step_fn, args=(next(iterator)))

            train_iterator = iter(train_data)

            for epoch in range(self.config.epochs):
                print(f'Epoch {epoch}')

                for step in range(10000):
                    train_step(train_iterator)
                    print(training_mean.result())
                    print(training_acc.result())
                print()


        else:
            print('No TPU Cluster, Train Via GPU\n')
            self.train_model()


if __name__ == '__main__':
    test_trainer = TrainTransformer(data_name='ted_hrlr_translate/pt_to_en',
                                    filter_length=100, buffer_size=20000, batch_size= 64)

    test_trainer.train_model()