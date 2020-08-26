import context

from datetime import datetime
import cv2
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS

from model import face_detection
from losses import SSDLosses
from dataset import prior_boxes
from dataset.wider_face import DataLoader
from config import cfg
from utils.box_utils import decode

devices = tf.config.experimental.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)


@tf.function
def train_step(images, gt_confs, gt_locs, model, loss_obj, optimizer):
    with tf.GradientTape() as tape:
        confs, locs = model(images)

        conf_loss, loc_loss = loss_obj(
            confs, locs, gt_confs, gt_locs)
        loss = conf_loss + loc_loss

        l2_loss = [tf.nn.l2_loss(t) for t in model.trainable_variables]
        l2_loss = 5e-4 * tf.math.reduce_sum(l2_loss)
        loss += l2_loss

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, conf_loss, loc_loss, l2_loss


loss_obj = SSDLosses(cfg['MODEL']['NEG_RATIO'],
                     cfg['MODEL']['NUM_CLASSES'])
image_size = cfg['INPUT']['IMAGE_SIZE']
model_name = cfg['MODEL']['NAME']
boxes_per_location = cfg['MODEL']['PRIORS']['BOXES_PER_LOCATION']
num_classes = cfg['MODEL']['NUM_CLASSES']
input_shape = (image_size, image_size, 3)
model = face_detection.FaceDetection(input_shape, num_classes,
                                     boxes_per_location, training=True)
# model = face_detection.create_model(cfg, training=True)
optimizer = tf.keras.optimizers.Adam(1e-3)

priors = prior_boxes.PriorBox(cfg).forward()
train_loader = DataLoader(priors, batch_size=16,
                          num_workers=6,
                          image_size=image_size,
                          training=True)
train_data = train_loader.load('./data/wider_face_train.tfrecord')

import time
for epoch in range(100):
    beg = time.time()
    for batch, (images, gt_confs, gt_locs) in enumerate(train_data):
        loss, conf_loss, loc_loss, l2_loss = train_step(
            images, gt_confs, gt_locs, model, loss_obj, optimizer
        )
        if batch % 20 == 0:
            d = datetime.fromtimestamp(time.time())
            d_str = d.strftime('%Y-%m-%d %H:%M:%S')
            print('{} {} {} {}'.format(d_str, loss.numpy(), conf_loss.numpy(),
                                            loc_loss.numpy()))
    end = time.time()
    print('Epoch time: {:.2f}'.format(end - beg))
