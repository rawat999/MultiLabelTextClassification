import os
import tensorflow as tf
from models import TextMultiLabeledClassifier
from preprocessing import CustomTokenizer
from utils.metrices import f1_score
import pandas as pd
from datetime import datetime
from pathlib import Path
import argparse
from utils.args import print_args
from utils.read_yaml import load_yaml
import joblib

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


def train(model, x_train, y_train, x_val, y_val, epochs=20, batch_size=128, lr=1e-4):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
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

    return x_train, y_train, x_valid, y_valid, tokenizer.max_length, len(tokenizer.tokenizer.word_index) + 1


def print_f1_score(y, y_pred):
    act_f1_score = f1_score(y_true=y[0], y_pred=y_pred[0])
    obj_f1_score = f1_score(y_true=y[1], y_pred=y_pred[1])
    loc_f1_score = f1_score(y_true=y[2], y_pred=y_pred[2])

    print("********* Validation F1 Score Report *********")
    print("F1 Score for 'action': {}".format(act_f1_score))
    print("F1 Score for 'object': {}".format(obj_f1_score))
    print("F1 Score for 'location': {}".format(loc_f1_score))
    print("******************** END ***********************")


def main(train_file, valid_file, configs, is_save_model=True):
    train_data = pd.read_csv(train_file,
                             usecols=configs['use_data_cols']
                             )
    val_data = pd.read_csv(valid_file,
                           usecols=configs['use_data_cols']
                           )

    x_train, y_train, x_valid, y_valid, max_len, unique_tokens = train_val_data(train_data=train_data,
                                                                                val_data=val_data)

    print("Building Model...")
    model = build_model(max_len=max_len,
                        unique_tokens=unique_tokens,
                        )
    model.summary()

    print("Model training start...")
    model, _ = train(model, x_train, y_train, x_valid, y_valid,
                     epochs=int(configs['epochs']),
                     batch_size=int(configs['batch_size']),
                     lr=float(configs['learning_rate'])
                     )
    print("Done, Model training.")

    print("Preditions...")
    y_preds = model.predict(x_valid)

    print_f1_score(y=y_valid, y_pred=y_preds)

    if is_save_model:
        print("Saving the model...")
        joblib.dump(model, "classifier.pkl")
    print("DONE.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the MultiLabel Classfier. You must have run train.py first.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                     )
    parser.add_argument("-d", "--data", type=Path, default="data/",
                        help="Path to the directory where training and validation csv files are present.")

    parser.add_argument("-c", "--config", type=Path, default="config.yml",
                        help="Path to the config file for training")

    parser.add_argument("-s", "--save", type=int, default=0,
                        help="Boolean value for saving the model")

    args = parser.parse_args()

    pwd = os.getcwd()
    if os.path.exists(os.path.join(pwd, args.data)):
        if os.path.exists(os.path.join(pwd, args.data, "train_data.csv")):
            training_file = os.path.join(pwd, args.data, "train_data.csv")
        else:
            raise Exception("train_data.csv is not exist in {} directory".format(args.data))
        if os.path.exists(os.path.join(pwd, args.data, "valid_data.csv")):
            val_file = os.path.join(pwd, args.data, "valid_data.csv")
        else:
            raise Exception("valid_data.csv is not exist in {} directory".format(args.data))
    else:
        raise Exception("{} directory is not exist".format(args.data))

    if os.path.exists(os.path.join(pwd, args.config)):
        print("yml: ", args.config)
        if str(args.config).endswith('.yml'):
            config = load_yaml(os.path.join(pwd, args.config))
        else:
            raise Exception("Given wrong extention file: {}".format(args.config))
    else:
        raise Exception("{} file is not exists".format(args.config))

    # Start model training
    print_args(args, parser)
    main(train_file=training_file,
         valid_file=val_file,
         configs=config,
         is_save_model=args.save)
