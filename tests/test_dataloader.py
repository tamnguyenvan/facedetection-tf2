import context

import os
import cv2
import tensorflow as tf

from config import cfg
from dataset import prior_boxes
from dataset import wider_face
from utils.box_utils import decode

tf.config.run_functions_eagerly(True)

def parse_fn(example):
    """
    """
    example_fmt = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64)
    }
    parsed_example = tf.io.parse_single_example(example, example_fmt)
    image = tf.image.decode_jpeg(parsed_example['image/encoded'], 3)
    image = tf.cast(image, tf.float32)
    height = tf.cast(parsed_example['image/height'], tf.int32)
    width = tf.cast(parsed_example['image/width'], tf.int32)
    image_shape = tf.stack([height, width, 3])
    bboxes = tf.stack(
        [tf.sparse.to_dense(parsed_example['image/object/bbox/xmin']),
         tf.sparse.to_dense(parsed_example['image/object/bbox/ymin']),
         tf.sparse.to_dense(parsed_example['image/object/bbox/xmax']),
         tf.sparse.to_dense(parsed_example['image/object/bbox/ymax'])],
        axis=1)
    labels = tf.cast(
        tf.sparse.to_dense(parsed_example['image/object/class/label']),
        tf.int64)
    return image, bboxes, labels

priors = prior_boxes.PriorBox(cfg).forward()
loader = wider_face.DataLoader(priors, batch_size=4, training=False)

curr_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.abspath(os.path.join(curr_dir, '../data'))
tfrecord_file = os.path.join(data_dir, 'wider_face_val.tfrecord')
data = loader.load(tfrecord_file)
# data = tf.data.TFRecordDataset(tfrecord_file).map(parse_fn).batch(1)

for batch in data.take(2):
    pass


import pdb
# pdb.set_trace()
image = batch[0][0]
labels = batch[1][0]
bboxes = batch[2][0]

image = image.numpy()
image *= 255.
image = image.astype('uint8')
h, w = image.shape[:2]

bboxes = decode(priors, bboxes).numpy()
labels = labels.numpy()
# pdb.set_trace()
for box, label in zip(bboxes, labels):
    if label > 0:
        # pdb.set_trace()
        b = list(map(lambda x: int(x * w), box))
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imshow('img', image)
cv2.waitKey(0)
