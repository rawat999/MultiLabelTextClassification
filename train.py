import tensorflow as tf
from models import TextMultiLabeledClassifier
from preprocessing import CustomTokenizer
from utils.metrices import f1_score
import pandas as pd
from datetime import datetime


logdir = "notebooks/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


def build_model(max_len,
                unique_tokens,
                embedding_size=50):
    inputs = tf.keras.layers.Input(shape=(max_len,), name='input_layer')
    tmlc = TextMultiLabeledClassifier(unique_tokens=unique_tokens,
                                      emb_size=embedding_size, )
    outputs = tmlc.call(inputs)
    return tf.keras.models.Model(inputs=inputs, outputs=outputs)


def train(model, x_train, y_train, x_val, y_val, epochs=20, batch_size=128):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-03)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', f1_score])
    training_history = model.fit(x=x_train,
                                 y=y_train,
                                 validation_data=(x_val, y_val),
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 callbacks=[tensorboard_callback]
                                 )
    return model, training_history


def get_label2idx(data, label_cols):
    label2idx = {}
    for col in label_cols:
        unique_labels = data[col].unique().tolist()
        if 'none' in unique_labels:
            nidx = unique_labels.index('none')
            unique_labels[0], unique_labels[nidx] = unique_labels[nidx], unique_labels[0]
        label2idx[col] = {v: i for i, v in enumerate(unique_labels)}
    return label2idx


def convert_label_str2int(x, label2idx):
    act_label = label2idx['action'][x['action']]
    obj_label = label2idx['object'][x['object']]
    loc_label = label2idx['location'][x['location']]
    return [act_label, obj_label, loc_label]


def train_val_data(train_data, val_data):
    # Custom tokenizer
    tokenizer = CustomTokenizer(training_texts=train_data['transcription'])
    tokenizer.train_tokenize()

    x_train = tokenizer.vectorize_input(texts=train_data['transcription'])
    x_valid = tokenizer.vectorize_input(texts=val_data['transcription'])

    # label-indexing
    label2idx = get_label2idx(train_data, label_cols=['action', 'object', 'location'])

    # convert all labels from string to integer
    # for training
    y_train_action = train_data['action'].apply(lambda x: label2idx['action'][x])
    y_train_object = train_data['object'].apply(lambda x: label2idx['object'][x])
    y_train_location = train_data['location'].apply(lambda x: label2idx['location'][x])
    y_train = [y_train_action, y_train_object, y_train_location]

    # for validation
    y_val_action = val_data['action'].apply(lambda x: label2idx['action'][x])
    y_val_object = val_data['object'].apply(lambda x: label2idx['object'][x])
    y_val_location = val_data['location'].apply(lambda x: label2idx['location'][x])

    y_valid = [y_val_action, y_val_object, y_val_location]

    return x_train, y_train, x_valid, y_valid, tokenizer.max_length, len(tokenizer.tokenizer.word_index)+1


def main():
    train_data = pd.read_csv("train_data.csv",
                             usecols=['transcription', 'action', 'object', 'location']
                             )
    val_data = pd.read_csv("valid_data.csv",
                           usecols=['transcription', 'action', 'object', 'location']
                           )

    x_train, y_train, x_valid, y_valid, max_len, unique_tokens = train_val_data(train_data=train_data,
                                                                                val_data=val_data)

    model = build_model(max_len=max_len,
                        unique_tokens=unique_tokens)
    model.summary()
    model, hist = train(model, x_train, y_train, x_valid, y_valid)

    y_preds = model.predict(x_valid)
    print(y_preds[0].shape)
    print(y_preds[1].shape)
    print(y_preds[2].shape)
    print(f1_score(y_true=y_valid[0], y_pred=y_preds[0]))
    print(f1_score(y_true=y_valid[1], y_pred=y_preds[1]))
    print(f1_score(y_true=y_valid[2], y_pred=y_preds[2]))


if __name__ == '__main__':
    main()
