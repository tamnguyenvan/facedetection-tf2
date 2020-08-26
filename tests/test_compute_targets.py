import context

import tensorflow as tf
from utils.box_utils import compute_targets
from config import cfg
from dataset import prior_boxes


priors = prior_boxes.PriorBox(cfg).forward()

bboxes = tf.constant([[0.1, 0.1, 0.2, 0.2],
                      [0.12, 0.12, 0.22, 0.22]], tf.float32)
labels = tf.constant([1, 1], tf.int64)

gt_confs, gt_locs = compute_targets(priors, bboxes, labels)
