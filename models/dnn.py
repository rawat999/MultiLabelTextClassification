import tensorflow as tf


class ActionClassifierBlock(tf.keras.layers.Layer):
    def __init__(self, name='action_block'):
        super(ActionClassifierBlock, self).__init__(name=name)
        self.layer_1 = tf.keras.layers.LSTM(units=128,
                                            # dropout=0.2,
                                            # recurrent_dropout=0.5,
                                            name='action_lstm'
                                            )
        self.layer_2 = tf.keras.layers.Dense(units=6,
                                             activation='softmax',
                                             name='action'
                                             )

    def call(self, inputs, *args, **kwargs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        return x

    def get_config(self):
        config = super(ActionClassifierBlock, self).get_config()
        config.update({'action_lstm': self.layer_1,
                       'action': self.layer_2
                       })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ObjectClassifierBlock(tf.keras.layers.Layer):
    def __init__(self, name='object_block'):
        super(ObjectClassifierBlock, self).__init__(name=name)
        self.layer_1 = tf.keras.layers.LSTM(units=128,
                                            # dropout=0.2,
                                            # recurrent_dropout=0.5,
                                            return_sequences=True,
                                            name='object_lstm_1'
                                            )
        self.layer_2 = tf.keras.layers.LSTM(units=32,
                                            # dropout=0.2,
                                            # recurrent_dropout=0.5,
                                            return_sequences=False,
                                            name='object_lstm_2'
                                            )
        self.layer_3 = tf.keras.layers.Dense(units=14,
                                             activation='softmax',
                                             name='object'
                                             )

    def call(self, inputs, *args, **kwargs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return x

    def get_config(self):
        config = super(ObjectClassifierBlock, self).get_config()
        config.update({'object_lstm_1': self.layer_1,
                       'object_lstm_2': self.layer_2,
                       'object': self.layer_3
                       })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class LocationClassifierBlock(tf.keras.layers.Layer):
    def __init__(self, name='location_block'):
        super(LocationClassifierBlock, self).__init__(name=name)
        self.layer_1 = tf.keras.layers.LSTM(units=64,
                                            # dropout=0.2,
                                            # recurrent_dropout=0.5,
                                            name='location_lstm'
                                            )
        self.layer_2 = tf.keras.layers.Dense(units=4,
                                             activation='softmax',
                                             name='location'
                                             )

    def call(self, inputs, *args, **kwargs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        return x

    def get_config(self):
        config = super(LocationClassifierBlock, self).get_config()
        config.update({'location_lstm': self.layer_1,
                       'location': self.layer_2
                       })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TextMultiLabeledClassifier(tf.keras.Model):
    def __init__(self, unique_tokens, emb_size, name='classifier'):
        super(TextMultiLabeledClassifier, self).__init__(name=name)
        # self.inputs = tf.keras.layers.Input(shape=(max_seq_len,))
        self.embedding = tf.keras.layers.Embedding(input_dim=unique_tokens,
                                                   output_dim=emb_size,
                                                   trainable=True,
                                                   name='embedding'
                                                   )
        # self.dropout = tf.keras.layers.Dropout(rate=0.2, name='dropout')
        self.action_block = ActionClassifierBlock()
        self.object_block = ObjectClassifierBlock()
        self.locat_block = LocationClassifierBlock()

    def call(self, inputs, *args, **kwargs):
        x = self.embedding(inputs)
        # x = self.dropout(x)
        x1 = self.action_block.call(x)
        x2 = self.object_block.call(x)
        x3 = self.locat_block.call(x)
        return [x1, x2, x3]

    def get_config(self):
        config = super(TextMultiLabeledClassifier, self).get_config()
        config.update({'embedding': self.embedding,
                       # 'dropout': self.dropout,
                       'action_block': self.action_block,
                       'object_block': self.object_block,
                       'location_block': self.locat_block
                       })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
