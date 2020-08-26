import context

import os
import cv2
import tensorflow as tf

from config import cfg
from dataset import prior_boxes
from dataset import wider_face
from utils.box_utils import decode
from utils import input_utils

tf.config.run_functions_eagerly(True)


def show_image_with_labels(image, bboxes, labels):
    """
    """
    image = image.numpy()
    image = image * 255.
    image = image.astype('uint8')
    bboxes = bboxes.numpy()
    labels = labels.numpy()

    h, w = image.shape[:2]
    for box, label in zip(bboxes, labels):
        if label > 0:
            # b = list(map(lambda x: int(x * w), box))
            b = list(map(int, [box[0] * w, box[1] * h, box[2] * w, box[3] * h]))
            cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('img', image)
    cv2.waitKey(0)


def show_image(image, bboxes):
    """
    """
    image = image.numpy() * 255.
    image = image.astype('uint8')
    bboxes = bboxes.numpy()

    h, w = image.shape[:2]
    for box in bboxes:
        # b = list(map(lambda x: int(x * w), box))
        b = list(map(int, [box[0] * w, box[1] * h, box[2] * w, box[3] * h]))
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)

    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('img', image)
    cv2.waitKey(0)


# def random_crop(image, bboxes, labels, min_offset=0., max_offset=0.9):
#     """
#     """
#     # Ensure cropped image has a 
#
#
#
# def random_flip_left_right(image, bboxes):
#     """
#     """
#     image_raw = tf.identity(image)
#     bboxes_raw = tf.identity(bboxes)
#     image = tf.image.flip_left_right(image)
#     bboxes = tf.stack([
#         1. - bboxes[:, 2],
#         bboxes[:, 1],
#         1. - bboxes[:, 0],
#         bboxes[:, 3]], axis=1)
#     return image, bboxes, image_raw, bboxes_raw


priors = prior_boxes.PriorBox(cfg).forward()
loader = wider_face.DataLoader(priors, batch_size=1)

curr_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.abspath(os.path.join(curr_dir, '../data'))
data_file = os.path.join(data_dir, 'wider_face_train.tfrecord')
data = loader.load(data_file)

for batch in data.take(1):
    pass


image = batch[0][0]
labels = batch[1][0]
bboxes = batch[2][0]
bboxes = decode(priors, bboxes)
import pdb
# pdb.set_trace()

# import pdb
# pdb.set_trace()
print(image.shape)
show_image_with_labels(image, bboxes, labels)
