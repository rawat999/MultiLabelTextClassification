import yaml


def load_yaml(file):
    with open(file, 'r') as fp:
        try:
            data = yaml.safe_load(fp)
        except yaml.YAMLError as e:
            print(e)
    return data
