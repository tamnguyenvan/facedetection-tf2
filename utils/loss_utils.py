"""
"""
import tensorflow as tf


def hard_negative_mining(loss, gt_confs, neg_ratio):
    """ Hard negative mining algorithm
        to pick up negative examples for back-propagation
        base on classification loss values
    Args:
        loss: list of classification losses of all default boxes (B, num_default)
        gt_confs: classification targets (B, num_default)
        neg_ratio: negative / positive ratio
    Returns:
        conf_loss: classification loss
        loc_loss: regression loss
    """
    # loss: B x N
    # gt_confs: B x N
    pos_idx = gt_confs > 0
    num_pos = tf.reduce_sum(tf.dtypes.cast(pos_idx, tf.int32), axis=1)
    num_neg = num_pos * neg_ratio

    rank = tf.argsort(loss, axis=1, direction='DESCENDING')
    rank = tf.argsort(rank, axis=1)
    neg_idx = rank < tf.expand_dims(num_neg, 1)

    return pos_idx, neg_idx


def smooth_L1_loss(y_true, y_pred):
    """
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    diff = tf.abs(y_true - y_pred)
    return tf.reduce_sum(
        tf.where(tf.less(diff, 1.), 0.5 * tf.square(diff), diff - 0.5))


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining.
    This will be used to determine unaveraged confidence loss across all
    examples in a batch.

    Args
      :x: A tensor of shape (N, num_classes) as conf_pred.
    """
    x_max = tf.reduce_max(x)
    return tf.math.log(tf.reduce_sum(tf.exp(x - x_max), 1)) + x_max
