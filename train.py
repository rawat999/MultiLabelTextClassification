import os
import tensorflow as tf
from models import build_model
from preprocessing import CustomTokenizer
from utils.metrices import f1_score, print_f1_score
import pandas as pd
from datetime import datetime
from pathlib import Path
import argparse
from utils.args import print_args
from utils.handle_yaml import load_yaml
from utils.label_map import get_label2idx
import joblib

logdir = "notebooks/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# checkpoint_path = "checkpoints/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                 save_weights_only=True,
#                                                 verbose=1)


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

    return x_train, y_train, x_valid, y_valid, tokenizer


def main(train_file, valid_file, configs, is_save_model=True):
    train_data = pd.read_csv(train_file,
                             usecols=configs['use_data_cols']
                             )
    val_data = pd.read_csv(valid_file,
                           usecols=configs['use_data_cols']
                           )

    x_train, y_train, x_valid, y_valid, tokenizer = train_val_data(train_data=train_data,
                                                                   val_data=val_data)

    print("Building Model...")
    model = build_model(max_len=tokenizer.max_length,
                        unique_tokens=len(tokenizer.tokenizer.word_index) + 1,
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
        print("Saving the tokenizer model...")
        joblib.dump(tokenizer, "tokenizer.pkl")
        print("Saving Model...")
        model.save("saved_model/tf_model")
    print("DONE.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the MultiLabel Classifier. "
                                                 "You must have run train.py first.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                     )
    parser.add_argument("-d", "--data", type=Path, default="data/",
                        help="Path to the directory where training and validation csv files are present.")

    parser.add_argument("-c", "--config", type=Path, default="config.yml",
                        help="Path to the config file for training")

    parser.add_argument("-s", "--save", type=int, default=1,
                        help="Boolean value for saving the model after training. "
                             "0 means not saving and 1 means save model")

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
