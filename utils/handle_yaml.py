import yaml


def load_yaml(file):
    with open(file, 'r') as fp:
        try:
            data = yaml.safe_load(fp)
        except yaml.YAMLError as e:
            print(e)
    return data


def write_yaml(file, d_map: dict):
    with open(file, "w") as fp:
        yaml.dump(d_map, fp)
