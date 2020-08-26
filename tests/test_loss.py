import context

import tensorflow as tf
from utils.box_utils import compute_targets
from config import cfg
from dataset import prior_boxes
from losses import SSDLosses


loss_obj = SSDLosses(cfg['MODEL']['NEG_RATIO'],
                     cfg['MODEL']['NUM_CLASSES'])
priors = prior_boxes.PriorBox(cfg).forward()

bboxes = tf.constant([[0.1, 0.1, 0.2, 0.2],
                      [0.12, 0.12, 0.22, 0.22]], tf.float32)
bboxes = tf.constant([[0.10546875, 0.59183675, 0.11914062, 0.6186896 ],
                      [0.9003906 , 0.5961332 , 0.91503906, 0.6165413 ],
                      [0.9238281 , 0.6100967 , 0.93847656, 0.632653  ],
                      [0.9814453 , 0.5950591 , 0.99609375, 0.6186896 ],
                      [0.8095703 , 0.62406015, 0.82421875, 0.64661646],
                      [0.87890625, 0.6262083 , 0.890625  , 0.6444683 ],
                      [0.96191406, 0.71213746, 0.9765625 , 0.7346939 ]],
                     dtype=tf.float32)
labels = tf.constant([1, 1, 1, 1, 1, 1, 1], tf.int64)

# import pdb
# pdb.set_trace()
gt_confs, gt_locs = compute_targets(priors, bboxes, labels)
gt_confs = tf.expand_dims(gt_confs, 0)
gt_locs = tf.expand_dims(gt_locs, 0)
confs = tf.constant([[[0.1, 0.9], [0.7, 0.5]]], tf.float32)
confs = tf.pad(confs, [[0, 0], [0, 5873], [0, 0]])
locs = tf.constant([[[0.1, 0.1, 0.15, 0.15],
                    [0.12, 0.12, 0.15, 0.15]]], tf.float32)
locs = tf.pad(locs, [[0, 0], [0, 5873], [0, 0]])
ret = loss_obj(confs, locs, gt_confs, gt_locs)
print(ret)
