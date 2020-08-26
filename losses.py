"""
"""
import tensorflow as tf

from utils.loss_utils import hard_negative_mining, log_sum_exp, smooth_L1_loss


class SSDLosses(object):
    """ Class for SSD Losses
    Attributes:
        neg_ratio: negative / positive ratio
        num_classes: number of classes
    """

    def __init__(self, neg_ratio, num_classes):
        self.neg_ratio = neg_ratio
        self.num_classes = num_classes

    def __call__(self, confs, locs, gt_confs, gt_locs):
        """ Compute losses for SSD
            regression loss: smooth L1
            classification loss: cross entropy
        Args:
            confs: outputs of classification heads (B, num_default, num_classes)
            locs: outputs of regression heads (B, num_default, 4)
            gt_confs: classification targets (B, num_default)
            gt_locs: regression targets (B, num_default, 4)
        Returns:
            conf_loss: classification loss
            loc_loss: regression loss
        """
        # regression loss only consist of positive examples
        # smooth_l1_loss = tf.keras.losses.Huber(reduction='sum')
        pos = gt_confs > 0
        pos_idx = tf.broadcast_to(
            tf.expand_dims(pos, -1), tf.shape(locs)
        )
        loc_p = tf.reshape(locs[pos_idx], (-1, 4))
        locs = tf.reshape(gt_locs[pos_idx], (-1, 4))
        loc_loss = smooth_L1_loss(loc_p, locs)

        # Compute classification losses without reduction
        # Compute a temporal loss for hard negative mining first
        batch_size = tf.shape(confs)[0]
        batch_conf = tf.reshape(confs, (-1, self.num_classes))
        sparse_conf_pred = tf.gather_nd(
            batch_conf,
            tf.stack([
                tf.range(tf.shape(batch_conf)[0], dtype=tf.int64),
                tf.cast(tf.reshape(gt_confs, (-1,)), tf.int64)
            ], axis=1)
        )
        temp_loss = log_sum_exp(batch_conf) - sparse_conf_pred

        # Filter out positive boxes for now
        indices = tf.where(tf.reshape(pos, (-1,)))
        temp_loss = tf.tensor_scatter_nd_update(
            temp_loss,
            indices,
            tf.zeros(tf.shape(indices)[0])
        )
        temp_loss = tf.reshape(temp_loss, (batch_size, -1))
        pos_idx, neg_idx = hard_negative_mining(
            temp_loss, gt_confs, self.neg_ratio)

        # classification loss will consist of positive and negative examples
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='sum')

        conf_loss = cross_entropy(
            gt_confs[tf.math.logical_or(pos_idx, neg_idx)],
            confs[tf.math.logical_or(pos_idx, neg_idx)])

        num_pos = tf.maximum(
            tf.reduce_sum(tf.dtypes.cast(pos_idx, tf.float32)), 1)

        conf_loss = conf_loss / num_pos
        loc_loss = loc_loss / num_pos

        return conf_loss, loc_loss
