import tensorflow as tf


def recall_m(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=1)
    conf_mat = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32)
    all_pos_count = tf.reduce_sum(conf_mat, axis=1)
    all_pos_count = tf.where(all_pos_count == 0, 1e-06, all_pos_count)
    recall = tf.divide(tf.linalg.diag_part(conf_mat), all_pos_count)
    return tf.reduce_mean(recall)


def precision_m(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=1)
    conf_mat = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32)
    all_pos_count = tf.reduce_sum(conf_mat, axis=0)
    all_pos_count = tf.where(all_pos_count == 0, 1e-06, all_pos_count)
    precision = tf.divide(tf.linalg.diag_part(conf_mat), all_pos_count)
    return tf.reduce_mean(precision)


def f1_score(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=1)
    conf_mat = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32)

    # Recall
    actual_pos_count = tf.reduce_sum(conf_mat, axis=1)
    actual_pos_count = tf.where(actual_pos_count == 0, 1e-9, actual_pos_count)
    recall = tf.divide(tf.linalg.diag_part(conf_mat), actual_pos_count)

    # Precision
    pred_pos_count = tf.reduce_sum(conf_mat, axis=0)
    pred_pos_count = tf.where(pred_pos_count == 0, 1e-9, pred_pos_count)
    precision = tf.divide(tf.linalg.diag_part(conf_mat), pred_pos_count)

    # F1-Score
    f1_measure = tf.math.divide((2 * tf.math.multiply(precision, recall)),
                                tf.math.add(precision, recall) + 1e-9)
    return tf.reduce_mean(f1_measure)
