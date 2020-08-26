"""
"""
import os
import cv2
import numpy as np
import tensorflow as tf

from utils import box_utils, input_utils


class DataLoader:
    """Abstract class for WIDER_FACE data loader"""
    def __init__(self, prior_boxes, batch_size=32, num_workers=None,
                 image_size=320, max_boxes=200, training=True):
        """
        Args
          :prior_boxes: The default boxes.
          :batch_size: The batch size.
          :num_workers: Number of parallel workers.
          :image_size: The input image size.
          :max_boxes: Maxium boxes per image.
          :training: Training phase.
        """
        self.prior_boxes = prior_boxes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.max_boxes = max_boxes
        self.training = training

    def _parse_fn(self, example):
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
        image = tf.cast(image, dtype=tf.float32)
        height = tf.cast(parsed_example['image/height'], tf.int32)
        width = tf.cast(parsed_example['image/width'], tf.int32)
        image_shape = tf.stack([height, width])
        bboxes = tf.stack(
            [tf.sparse.to_dense(parsed_example['image/object/bbox/xmin']),
             tf.sparse.to_dense(parsed_example['image/object/bbox/ymin']),
             tf.sparse.to_dense(parsed_example['image/object/bbox/xmax']),
             tf.sparse.to_dense(parsed_example['image/object/bbox/ymax'])],
            axis=1)
        labels = tf.cast(
            tf.sparse.to_dense(parsed_example['image/object/class/label']),
            tf.int64)

        # Dta augmentation
        image, image_shape, bboxes = tf.cond(
            tf.equal(self.training, True),
            lambda: self._transform(image, image_shape, bboxes),
            lambda: (image / 255., image_shape, bboxes)
        )

        # Resize to fixed-size image
        image, bboxes = input_utils.resize_with_bboxes(
            image, image_shape, bboxes, self.image_size, self.image_size
        )

        # Transform targets
        gt_confs, gt_locs = box_utils.compute_targets(
            self.prior_boxes, bboxes, labels)

        return image, gt_confs, gt_locs

    def _transform(self, image, image_shape, bboxes):
        """Perform data augmentation.

        Args
          :image: 3D tensor of shape (H, W, C).
          :image_shape: 2-value tensor specify image shape.
          :bboxes: 2D tensor of shape (num_boxes, 4) containing bounding boxes
          in format (xmin, ymin, xmax, ymax).

        Returns
          :image: The distorted image.
          :image_shape: The new image shape if any changed.
          :bboxes: The new bounding boxes coordinates if any changed.
        """
        # Do image augmentation here
        bboxes = tf.clip_by_value(bboxes, 0., 1.)
        image, bboxes = input_utils.random_flip_left_right(
            image, bboxes)
        image = input_utils.random_erasing(image, image_shape, max_area=0.01)
        image, image_shape, bboxes = input_utils.random_crop_with_bboxes(
            image, image_shape,  bboxes)
        image /= 255.
        return image, image_shape, bboxes

    def _transform_batch(self, images, gt_confs, gt_locs):
        """Perform augmentation by batch."""
        images = input_utils.random_brightness(images, 0.3)
        images = input_utils.random_hue(images, 0.1)
        return images, gt_confs, gt_locs

    def load(self, tfrecord_file):
        """Load data and create tf dataset.

        Args
          :data_dir:

        Returns
          :dataset: A tf dataset object.
          :length: Length of the dataset.
        """
        dataset = tf.data.TFRecordDataset(tfrecord_file)
        if self.training:
            dataset = dataset.shuffle(1000)

        AUTO = tf.data.experimental.AUTOTUNE
        if not isinstance(self.num_workers, int):
            self.num_workers = AUTO
        dataset = dataset.map(
            self._parse_fn, num_parallel_calls=self.num_workers)

        if self.training:
            dataset = dataset.map(
                self._transform_batch, num_parallel_calls=self.num_workers)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(AUTO)
        return dataset
