import tensorflow as tf

def f1_score(y_true, y_pred):
    """Calculate the F1 score."""
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(tf.round(y_pred), 'float32')
    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)
    precision = tp / (tp + fp + 1e-16)
    recall = tp / (tp + fn + 1e-16)
    f1 = 2 * precision * recall / (precision + recall + 1e-16)
    return tf.reduce_mean(f1)
