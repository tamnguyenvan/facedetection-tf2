"""Create tfrecord files for training."""
import os
import json
from absl import app, flags, logging
from absl.flags import FLAGS

import cv2
import tqdm
import numpy as np
import tensorflow as tf

WIDER_CLASSES = ('__background__', 'face')
TRAIN_IMAGE_DIR = 'WIDER_train/images'
VAL_IMAGE_DIR = 'WIDER_val/images'
ANNO_DIR = 'wider_face_split'

flags.DEFINE_enum('split', 'train', ['train', 'val'], 'Train or val')
flags.DEFINE_string('data_dir', 'data', 'Data dir')
flags.DEFINE_string('out_file', 'wider_face.tfrecord', 'Output file')


def bytes_feature(value):
    if not isinstance(value, (tuple, list)):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    if not isinstance(value, (tuple, list)):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
    if not isinstance(value, (tuple, list)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def build_example(image_path, ann_data):
    """Build an example from annotation data."""
    filename = os.path.basename(image_path)
    img_raw = open(image_path, 'rb').read()

    width = ann_data['width']
    height = ann_data['height']
    bboxes = ann_data['bboxes']
    class_ids = ann_data['classes']

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    for bbox, class_id in zip(bboxes, class_ids):
        x, y, w, h = bbox
        xmin.append(float(x) / width)
        ymin.append(float(y) / height)
        xmax.append(float(x + w) / width)
        ymax.append(float(y + h) / height)
        classes.append(class_id)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(value=height),
        'image/width': int64_feature(value=width),
        'image/filename': bytes_feature(value=filename.encode('utf-8')),
        'image/encoded': bytes_feature(value=img_raw),
        'image/object/bbox/xmin': float_feature(value=xmin),
        'image/object/bbox/ymin': float_feature(value=ymin),
        'image/object/bbox/xmax': float_feature(value=xmax),
        'image/object/bbox/ymax': float_feature(value=ymax),
        'image/object/class/label': int64_feature(value=classes)
    }))
    return example


def load_data(data_dir, split):
    if split == 'train':
        image_dir = os.path.join(data_dir, TRAIN_IMAGE_DIR)
        anno_path = os.path.join(
            data_dir, ANNO_DIR, 'wider_face_train_bbx_gt.txt')
    elif split == 'val':
        image_dir = os.path.join(data_dir, VAL_IMAGE_DIR)
        anno_path = os.path.join(
            data_dir, ANNO_DIR, 'wider_face_val_bbx_gt.txt')

    anns = {}
    with open(anno_path, 'rt') as f:
        for line in tqdm.tqdm(f):
            line = line.strip()
            if line.endswith('.jpg'):
                image_path = os.path.join(image_dir, line)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                height, width = image.shape[:2]
                anns[image_path] = {
                    'width': width,
                    'height': height,
                    'bboxes': [],
                    'classes': []
                }
            else:
                split_line = line.split()
                if len(split_line) > 1:
                    x, y, w, h = list(map(int, split_line[:4]))
                    invalid = int(split_line[7])
                    if invalid == 1:
                        continue
                    anns[image_path]['bboxes'].append((x, y, w, h))
                    anns[image_path]['classes'].append(1)
    return anns


def main(_argv):
    logging.info(f'Loading data from {FLAGS.data_dir}')
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(root_dir, FLAGS.data_dir)
    anns = load_data(data_dir, FLAGS.split)

    logging.info(f'Loaded {len(anns)} images')
    writer = tf.io.TFRecordWriter(FLAGS.out_file)
    for image_path, ann_data in anns.items():
        tf_example = build_example(image_path, ann_data)
        writer.write(tf_example.SerializeToString())
    writer.close()
    logging.info(f'Saved as {FLAGS.out_file}')


if __name__ == '__main__':
    app.run(main)
