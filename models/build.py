import tensorflow as tf
from models import TextMultiLabeledClassifier


def build_model(max_len,
                unique_tokens,
                embedding_size=50):
    inputs = tf.keras.layers.Input(shape=(max_len,), name='input_layer')
    tmlc = TextMultiLabeledClassifier(unique_tokens=unique_tokens,
                                      emb_size=embedding_size, )
    outputs = tmlc.call(inputs)
    return tf.keras.models.Model(inputs=inputs, outputs=outputs)
