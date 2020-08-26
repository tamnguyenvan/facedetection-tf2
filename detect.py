"""
"""
import os
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
flags.DEFINE_string('image', '', 'Path to input file')
flags.DEFINE_float('conf_threshold', 0.3,
                   'Ignore boxes whose scores are too low')
flags.DEFINE_float('nms_threshold', 0.3, 'NMS threshold')
flags.DEFINE_integer('keep_top_k', 750, 'Keep top-k boxes')
flags.DEFINE_integer('top_k', 5000, 'top-k boxes')


def main(_argv):
    """
    """
    devices = tf.config.experimental.list_physical_devices('GPU')
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)

    # # Load the model
    # image_size = cfg['INPUT']['IMAGE_SIZE']
    # model_name = cfg['MODEL']['NAME']
    # boxes_per_location = cfg['MODEL']['PRIORS']['BOXES_PER_LOCATION']
    # num_classes = cfg['MODEL']['NUM_CLASSES']
    # neg_ratio = cfg['MODEL']['NEG_RATIO']
    # num_classes = cfg['MODEL']['NUM_CLASSES']
    # input_shape = (image_size, image_size, 3)
    # model = face_detection.FaceDetection(input_shape, num_classes,
    #                                      boxes_per_location, training=False)
    # model_name = cfg['MODEL']['NAME']
    # logging.info(f'Loaded model {model_name}')
    #
    # optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    #
    # # Create a checkpoint for smooth training
    # ckpt = tf.train.Checkpoint(
    #     step=tf.Variable(0), optimizer=optimizer, model=model)
    # manager = tf.train.CheckpointManager(ckpt, FLAGS.model, max_to_keep=1)
    #
    # # Retore variables if checkpoint exists
    # ckpt.restore(manager.latest_checkpoint)
    # if manager.latest_checkpoint:
    #     logging.info('Restoring from {}'.format(manager.latest_checkpoint))
    # else:
    #     logging.info('Train the model from scratch')
    model = tf.keras.models.load_model('checkpoints/saved_models')

    image_path = FLAGS.image
    image_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    h, w = image_raw.shape[:2]
    image = cv2.resize(image_raw, (320, 320))
    image = image.astype('float32') / 255.
    images = np.expand_dims(image, axis=0)

    priors = prior_boxes.PriorBox(cfg).forward()
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
    for b in dets:
        score = float(b[4])
        text = 'score: {:.2f}'.format(score)
        b = list(map(int, b[:4]))
        cv2.rectangle(image_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)
        cx = b[0]
        cy = b[1] - 12
        cv2.putText(image_raw, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (12, 215, 255))
    cv2.imshow('img', image_raw)
    cv2.waitKey(0)


if __name__ == '__main__':
    app.run(main)
