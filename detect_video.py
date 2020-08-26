"""
"""
import os
import time
import cv2
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf

from config import cfg
from dataset import prior_boxes
from model import face_detection
from utils.box_utils import decode
from utils.nms import nms


flags.DEFINE_string('model', '', 'Path to model checkpoint')
flags.DEFINE_float('conf_threshold', 0.3,
                   'Ignore boxes whose scores are too low')
flags.DEFINE_float('nms_threshold', 0.3, 'NMS threshold')
flags.DEFINE_integer('keep_top_k', 750, 'Keep top-k boxes')
flags.DEFINE_integer('top_k', 5000, 'top-k boxes')
flags.DEFINE_integer('device', 'gpu', 'The runtime device')


IMAGE_SIZE = cfg['INPUT']['IMAGE_SIZE']


def detect(image, model, priors):
    """
    """
    h, w = image.shape[:2]
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.astype('float32')
    images = np.expand_dims(image, axis=0)

    confs, locs = model(images, training=False)
    boxes = decode(priors, tf.squeeze(locs, 0))
    boxes = boxes.numpy()
    scale = np.array([w, h, w, h])
    boxes = boxes * scale

    confs = tf.squeeze(confs, 0)
    scores = confs.numpy()
    scores = scores[:, 1]

    # Ignore low scores
    inds = np.where(scores > FLAGS.conf_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # Keep top-k before NMS
    order = scores.argsort()[::-1][:FLAGS.top_k]
    boxes = boxes[order]
    scores = scores[order]

    # NMS
    dets = np.hstack(
        (boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    selected_idx = np.array([0, 1, 2, 3, 4])
    keep = nms(dets[:, selected_idx], FLAGS.nms_threshold)
    dets = dets[keep, :]

    dets = dets[:FLAGS.keep_top_k, :]
    return dets


def draw_bboxes(image, dets):
    """
    """
    for b in dets:
        score = float(b[4])
        text = 'score: {:.2f}'.format(score)
        b = list(map(int, b[:4]))
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)
        cx = b[0]
        cy = b[1] - 12
        cv2.putText(image, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (12, 215, 255))
    return image


def main(_argv):
    """
    """
    runtime_device = FLAGS.device.lower()
    if runtime_device == 'gpu':
        devices = tf.config.experimental.list_physical_devices('GPU')
        for device in devices:
            tf.config.experimental.set_memory_growth(device, True)
    else:
        devices = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices=devices,
                                                   device_type='CPU')

    priors = prior_boxes.PriorBox(cfg).forward()
    model = tf.keras.models.load_model('checkpoints/saved_models')

    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()
        if not ret:
            continue

        start = time.time()
        dets = detect(image, model, priors)
        end = time.time()
        print('Detect time: {:.2f} {}FPS'.format(end - start,
                                                 int(1 / (end - start))))

        drawn_image = draw_bboxes(image, dets)
        cv2.imshow('face detection', drawn_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    app.run(main)
