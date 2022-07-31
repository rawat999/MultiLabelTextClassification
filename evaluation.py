import os
import argparse
import joblib
from pathlib import Path
import pandas as pd
from utils.handle_yaml import load_yaml
from utils.metrices import print_f1_score
from utils.args import print_args


def process_data(test_data, tokenizer_file, label_map_file):
    # Load tokenizer
    tokenizer = joblib.load(tokenizer_file)

    x_test = tokenizer.vectorize_input(texts=test_data['transcription'])

    # label-indexing
    label2idx = load_yaml(label_map_file)

    # convert all labels from string to integer
    # for testing
    y_test_action = test_data['action'].apply(lambda x: label2idx['action'][x])
    y_test_object = test_data['object'].apply(lambda x: label2idx['object'][x])
    y_test_location = test_data['location'].apply(lambda x: label2idx['location'][x])
    y_test = [y_test_action, y_test_object, y_test_location]

    return x_test, y_test


def main(test_file, model_file, tokenizer_file, label_map_file, configs):
    # data processing
    test_data = pd.read_csv(test_file,
                            usecols=configs['use_data_cols'])
    x, y = process_data(test_data=test_data,
                        tokenizer_file=tokenizer_file,
                        label_map_file=label_map_file)

    # model loading
    model = joblib.load(model_file)

    # model prediction
    y_preds = model.predict(x)

    # print f1 scores
    print_f1_score(y=y, y_pred=y_preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the MultiLabel Classfier. "
                                                 "You must have run train.py before evaluate.py",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                     )
    parser.add_argument("-d", "--data", type=Path, default="test.csv",
                        help="Path to the file where test csv file is present.")

    parser.add_argument("-m", "--model", type=Path, default="classifier.pkl",
                        help="Path to the model file where pkl file is present.")

    parser.add_argument("-t", "--tokenizer", type=Path, default="tokenizer.pkl",
                        help="Path to the tokenizer file where pkl file is present.")

    parser.add_argument("-l", "--labels", type=Path, default="label_mapping.yml",
                        help="Path to the label to index mapping file where yml file is present.")

    parser.add_argument("-c", "--config", type=Path, default="config.yml",
                        help="Path to the config file for training")

    args = parser.parse_args()

    pwd = os.getcwd()

    # test data file
    if os.path.exists(os.path.join(pwd, args.data)):
        testing_file = os.path.join(pwd, args.data)
    else:
        raise Exception("{} is not exists".format(args.data))

    # model file
    if os.path.exists(os.path.join(pwd, args.model)):
        mod_file = os.path.join(pwd, args.model)
    else:
        raise Exception("{} is not exists".format(args.model))

    # tokenizer file
    if os.path.exists(os.path.join(pwd, args.tokenizer)):
        tok_file = os.path.join(pwd, args.tokenizer)
    else:
        raise Exception("{} is not exists".format(args.tokenizer))

    # label mapping file
    if os.path.exists(os.path.join(pwd, args.labels)):
        lab_file = os.path.join(pwd, args.labels)
    else:
        raise Exception("{} is not exists".format(args.labels))

    # config file
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
    main(test_file=testing_file,
         model_file=mod_file,
         tokenizer_file=tok_file,
         label_map_file=lab_file,
         configs=config
         )
