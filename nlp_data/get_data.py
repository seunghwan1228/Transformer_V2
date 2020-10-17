import tensorflow_datasets as tfds


# Get Data
class GetData:
    def __init__(self, data_name):
        self.data_name = data_name

    def get_data(self):
        examples, metadata = tfds.load(self.data_name, with_info=True, as_supervised=True)
        return examples, metadata
