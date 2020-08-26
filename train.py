"""Train the model"""
import os
import time

import yaml
from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf

from model import face_detection
from losses import SSDLosses
from dataset import prior_boxes
from dataset.wider_face import DataLoader
from metrics.average_precision import eval_detection_voc
from config import cfg


flags.DEFINE_string('train_file', 'data/wider_face_train.tfrecord',
                    'Path to training tfrecord file')
flags.DEFINE_string('val_file', 'data/wider_face_val.tfrecord',
                    'Path to validation tfrecord file')
flags.DEFINE_integer('batch_size', 16, 'Batch size')
flags.DEFINE_integer('epoch', 500, 'Number of epochs')
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_float('momentum', 0.9, 'SGD momentum')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight decay')
flags.DEFINE_integer('num_workers', 8, 'Number of data loading threads')
flags.DEFINE_integer('log_step', 50, 'Print logs every `log_step`')
flags.DEFINE_integer('eval_step', 5, 'Evaluate model every `eval_step`')
flags.DEFINE_integer('patience', 5,
                     'Early stopping after `patience` epochs')
flags.DEFINE_string('checkpoint_prefix', 'checkpoints',
                    'Checkpoint prefix')
flags.DEFINE_string('saved_models', 'checkpoints/saved_models',
                    'Directory to save the final model in SavedModel format')


@tf.function
def train_step(images, gt_confs, gt_locs, model, loss_obj, optimizer):
    with tf.GradientTape() as tape:
        confs, locs = model(images)

        conf_loss, loc_loss = loss_obj(
            confs, locs, gt_confs, gt_locs)
        loss = conf_loss + loc_loss

        l2_loss = [tf.nn.l2_loss(t) for t in model.trainable_variables]
        l2_loss = FLAGS.weight_decay * tf.math.reduce_sum(l2_loss)
        loss += l2_loss

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, conf_loss, loc_loss, l2_loss


@tf.function
def test_step(images, gt_confs, gt_locs, model, loss_obj):
    confs, locs = model(images)

    conf_loss, loc_loss = loss_obj(
        confs, locs, gt_confs, gt_locs)
    loss = conf_loss + loc_loss

    l2_loss = [tf.nn.l2_loss(t) for t in model.trainable_variables]
    l2_loss = FLAGS.weight_decay * tf.math.reduce_sum(l2_loss)
    loss += l2_loss

    return loss, conf_loss, loc_loss, l2_loss


def main(_argv):
    devices = tf.config.experimental.list_physical_devices('GPU')
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)

    # tf.config.run_functions_eagerly(True)

    image_size = cfg['INPUT']['IMAGE_SIZE']
    model_name = cfg['MODEL']['NAME']
    boxes_per_location = cfg['MODEL']['PRIORS']['BOXES_PER_LOCATION']
    num_classes = cfg['MODEL']['NUM_CLASSES']
    neg_ratio = cfg['MODEL']['NEG_RATIO']
    num_classes = cfg['MODEL']['NUM_CLASSES']

    # Build the model
    input_shape = (image_size, image_size, 3)
    model = face_detection.FaceDetection(input_shape, num_classes,
                                         boxes_per_location, training=True)
    logging.info(f'Build model {model_name}')

    # Optimizer and loss object
    loss_obj = SSDLosses(neg_ratio, num_classes)
    optimizer = tf.keras.optimizers.Adam(FLAGS.lr)
    # optimizer = tf.keras.optimizers.SGD(FLAGS.lr)

    # Checkpoint and checkpoint manager
    ckpt = tf.train.Checkpoint(step=tf.Variable(0),
                               model=model,
                               best_loss=tf.Variable(1e6),
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                              FLAGS.checkpoint_prefix,
                                              max_to_keep=3)
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    if ckpt_manager.latest_checkpoint:
        logging.info('Restored model from {}'.format(
            ckpt_manager.latest_checkpoint))
    else:
        logging.info('Training the model from scratch')

    # Load data
    priors = prior_boxes.PriorBox(cfg).forward()
    train_loader = DataLoader(priors, batch_size=FLAGS.batch_size,
                              num_workers=FLAGS.num_workers,
                              image_size=image_size,
                              training=True)
    train_data = train_loader.load(FLAGS.train_file)
    logging.info(f'Loaded train data from {FLAGS.train_file}')

    val_loader = DataLoader(priors, batch_size=FLAGS.batch_size,
                            num_workers=FLAGS.num_workers,
                            image_size=image_size,
                            training=False)
    val_data = val_loader.load(FLAGS.val_file)
    logging.info(f'Loaded validation data from {FLAGS.val_file}')
    
    # Loss aggregation
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_conf_loss = tf.keras.metrics.Mean(name='train_conf_loss')
    train_loc_loss = tf.keras.metrics.Mean(name='train_loc_loss')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_conf_loss = tf.keras.metrics.Mean(name='val_conf_loss')
    val_loc_loss = tf.keras.metrics.Mean(name='val_loc_loss')

    # Tensorboard summaries
    train_log_dir = 'logs/train'
    val_log_dir = 'logs/val'
    train_writer = tf.summary.create_file_writer(train_log_dir)
    val_writer = tf.summary.create_file_writer(val_log_dir)

    # patience_cnt = 0
    for epoch in range(int(ckpt.step) + 1, FLAGS.epoch + 1):
        start = time.time()
        for batch, (images, gt_confs, gt_locs) in enumerate(train_data):
            loss, conf_loss, loc_loss, l2_loss = train_step(
                images, gt_confs, gt_locs, model, loss_obj, optimizer
            )
            train_loss.update_state(loss)
            train_conf_loss.update_state(conf_loss)
            train_loc_loss.update_state(loc_loss)

            if (batch + 1) % FLAGS.log_step == 0:
                logging.info('Epoch {} iter {} | conf_loss: {:.2f} '
                             ' loc_loss: {:.2f} l2_loss: {:.2f} '
                             ' loss: {:.2f}'.format(
                                 epoch, batch + 1,
                                 conf_loss.numpy(),
                                 loc_loss.numpy(),
                                 l2_loss.numpy(),
                                 loss.numpy()))
        end = time.time()
        logging.info('Epoch time: {:.2f}s'.format(end - start))

        # Trainning summaries
        with train_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('conf_loss', train_conf_loss.result(), step=epoch)
            tf.summary.scalar('loc_loss', train_loc_loss.result(), step=epoch)

        train_loss.reset_states()
        train_conf_loss.reset_states()
        train_loc_loss.reset_states()

        # if epoch % FLAGS.eval_step == 0:
        for batch, (images, gt_confs, gt_locs) in enumerate(val_data):
            loss, conf_loss, loc_loss, _ = test_step(
                images, gt_confs, gt_locs, model, loss_obj
            )
            val_loss.update_state(loss)
            val_conf_loss.update_state(conf_loss)
            val_loc_loss.update_state(loc_loss)

        logging.info('Evaluation | conf_loss: {:.2f} loc_loss {:.2f} '
                     'loss: {:.2f} best_loss: {:.2f}'.format(
                         val_conf_loss.result(),
                         val_loc_loss.result(),
                         val_loss.result(),
                         float(ckpt.best_loss)))

        with val_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=epoch)
            tf.summary.scalar('conf_loss', val_conf_loss.result(), step=epoch)
            tf.summary.scalar('loc_loss', val_loc_loss.result(), step=epoch)

        # Save checkpoint
        save_path = ckpt_manager.save()
        logging.info(f'Saved checkpoint as {save_path}')
        if val_loss.result() <= float(ckpt.best_loss):
            # Save best model in SavedModel format
            model.save(FLAGS.saved_models)
            ckpt.best_loss.assign(val_loss.result())
            logging.info(f'Saved best model in {FLAGS.saved_models}')

        val_loss.reset_states()
        val_conf_loss.reset_states()
        val_loc_loss.reset_states()

        # Increase epoch counter
        ckpt.step.assign_add(1)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
