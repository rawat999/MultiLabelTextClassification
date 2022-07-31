from utils.handle_yaml import write_yaml


def get_label2idx(data, label_cols):
    label2idx = {}
    for col in label_cols:
        unique_labels = data[col].unique().tolist()
        if 'none' in unique_labels:
            nidx = unique_labels.index('none')
            unique_labels[0], unique_labels[nidx] = unique_labels[nidx], unique_labels[0]
        label2idx[col] = {v: i for i, v in enumerate(unique_labels)}
    write_yaml("label_mapping.yml", d_map=label2idx)
    return label2idx
