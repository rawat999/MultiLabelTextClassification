import tensorflow as tf

MAX_SEQUENCE_LENGTH = 24


class CustomTokenizer:
    def __init__(self, training_texts):
        self.train_texts = training_texts
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=1000)
        self.max_length = None

    def train_tokenize(self):
        max_length = len(max(self.train_texts, key=len))
        self.max_length = min(max_length, MAX_SEQUENCE_LENGTH)
        self.tokenizer.fit_on_texts(self.train_texts)

    def vectorize_input(self, texts):
        texts = self.tokenizer.texts_to_sequences(texts)
        texts = tf.keras.preprocessing.sequence.pad_sequences(sequences=texts,
                                                              maxlen=self.max_length,
                                                              truncating='post',
                                                              padding='post')
        return texts
