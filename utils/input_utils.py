"""
"""
import tensorflow as tf


def _flip_left_right(image, bboxes):
    """Flip the given image and bounding boxes horizontally (left to right).

    Args
      :image: 3D tensor of shape (H, W, C).
      :bboxes: 2D tensor of shape (num_boxes, 4) containing bounding boxes
      in format (xmin, ymin, xmax, ymax)

    Returns
      :image: New image was flipped.
      :bboxes: New bounding boxes corresponding.
    """
    image = tf.image.flip_left_right(image)
    bboxes = tf.stack([
        1. - bboxes[:, 2],
        bboxes[:, 1],
        1. - bboxes[:, 0],
        bboxes[:, 3]], axis=1)
    return image, bboxes


@tf.function
def random_flip_left_right(image, bboxes):
    """Flip the given image and bounding boxes from left to right randomly.

    Args
      :image: 3D tensor of shape (H, W, C).
      :bboxes: 2D tensor of shape (num_boxes, 4) containing bounding boxes
      in format (xmin, ymin, xmax, ymax)

    Returns
      :image: New image was flipped.
      :bboxes: New bounding boxes corresponding.
    """
    image, bboxes = tf.cond(
        tf.less(tf.random.uniform([], 0, 1), 0.5),
        lambda: (image, bboxes),
        lambda: _flip_left_right(image, bboxes)
    )
    return image, bboxes


def _crop_bboxes(bboxes, left, top, new_width, new_height):
    """Crop bboxes with respect top-left coordinate and size.

    Args
      :bboxes: 2D tensor of shape (x1, y1, x2, y2) containing bounding boxes in
      format (xmin, ymin, xmax, ymax).
      :left: The x-axis value of the left most point should be in range [0, 1].
      :top: The y-axis value of the top most point should be in range [0, 1].
      :new_width: The new width of cropped area in range [0, 1].
      :new_height: The new height of cropped area in range [0, 1].

    Returns
      :bboxes: The bboxes with new coodinates.
    """
    # move the coordinates according to new min value
    top_left = tf.stack([left, top, left, top])
    bboxes = bboxes - top_left

    scale = [new_width, new_height, new_width, new_height]
    bboxes = bboxes / scale
    return bboxes


def _crop_with_bboxes(image, image_shape, bboxes):
    """Crop image with all bounding boxes reserved.

    Args
      :image: 3D tensor of shape (H, W, C).
      :bboxes: 2D tensor of shape (num_boxes, 4) containing bounding boxes in
      format (xmin, ymin, xmax, ymax).

    Returns
      :image: New image was cropped
      :bboxes: New bounding boxes respectly.
    """
    height, width = tf.unstack(tf.cast(image_shape[:2], tf.float32))
    # height, width = tf.unstack(image_shape)

    # Calculate coordinates
    x1 = tf.random.uniform([], 0., tf.reduce_min(bboxes[:, 0]))
    y1 = tf.random.uniform([], 0., tf.reduce_min(bboxes[:, 1]))
    x2 = tf.random.uniform([], tf.reduce_max(bboxes[:, 2]), 1.)
    y2 = tf.random.uniform([], tf.reduce_max(bboxes[:, 3]), 1.)
    new_height, new_width = y2 - y1, x2 - x1

    crop_bboxes_args = [x1, y1, new_width, new_height]
    target_h = tf.cast(new_height * height, dtype=tf.int32)
    target_w = tf.cast(new_width * width, dtype=tf.int32)
    crop_image_args = [
        tf.cast(y1 * height, dtype=tf.int32),
        tf.cast(x1 * width, dtype=tf.int32),
        target_h,
        target_w
    ]

    image, bboxes = tf.cond(
        tf.logical_and(tf.math.greater(target_h, 0),
                       tf.math.greater(target_w, 0)),
        lambda: (tf.image.crop_to_bounding_box(image, *crop_image_args),
                 _crop_bboxes(bboxes, *crop_bboxes_args)),
        lambda: (image, bboxes)
    )
    target_shape = tf.stack([target_h, target_w])
    return image, target_shape, bboxes


def random_crop_with_bboxes(image, image_shape, bboxes):
    """Randomly crop image with all bounding boxes reserved.

    Args
      :image: 3D tensor of shape (H, W, C).
      :bboxes: 2D tensor of shape (num_boxes, 4) containing bounding boxes in
      format (xmin, ymin, xmax, ymax).

    Returns
      :image: New image was cropped
      :bboxes: New bounding boxes respectly.
    """
    image, image_shape, bboxes = tf.cond(
        tf.less(tf.random.uniform([], 0, 1), 0.5),
        lambda: (image, image_shape, bboxes),
        lambda: _crop_with_bboxes(image, image_shape, bboxes)
    )
    return image, image_shape, bboxes


def random_erasing(image, image_shape, max_area=0.05, erased_value=0):
    """Randomly erase a rectangular part of the given image.

    Args
      :image: 3D tensor of shape (H, W, C).
      :max_area: Maximum part of image woulde be erased in range [0 1].
      :erased_value: The value which will be filled in the empty area.

    Returns
      :image: The new image with a part which be erased.
    """
    # import pdb
    # image_height, image_width = tf.unstack(tf.shape(image)[:2])
    image_height, image_width = tf.unstack(image_shape[:2])
    image_area = tf.cast(image_height * image_width, tf.float32)
    max_area = max_area * image_area

    # Chose width and height of erased area.
    erased_width = tf.cast(
        tf.sqrt(max_area / tf.random.uniform([], 0.5, 2.)), tf.int32)
    erased_height = tf.cast(
        max_area / tf.cast(erased_width, tf.float32), tf.int32)

    # Ensure reasonable width and height of the erased area.
    if tf.math.logical_or(tf.less(erased_width, 4),
                          tf.less(erased_height, 4)):
        return image

    # Get center of erased area
    center_x = tf.random.uniform([], erased_width // 2,
                                 image_width - erased_width // 2,
                                 dtype=tf.int32)
    center_y = tf.random.uniform([], erased_height // 2,
                                 image_height - erased_height // 2,
                                 dtype=tf.int32)

    # Calculate top-left bottom-right
    left = tf.maximum(center_x - erased_width // 2, 0)
    top = tf.maximum(center_y - erased_height // 2, 0)
    right = tf.minimum(left + erased_width, image_width)
    bottom = tf.minimum(top + erased_height, image_height)

    mask = tf.ones((erased_height, erased_width, 3), dtype=image.dtype)
    mask = tf.pad(mask, [[top, image_height - bottom],
                         [left, image_width - right],
                         [0, 0]])

    # Erase the image
    image = image * (1 - mask) + mask * erased_value
    return image


@tf.function
def resize_with_bboxes(image, image_shape, bboxes, target_w, target_h):
    """Resize image and scale bounding boxes also to new size.

    Args
      :image: 3D tensor of shape (H, W, C).
      :image_shape: 2-value tensor [image_height, image_width].
      :bboxes: 2D tensor of shape (num_boxes, 4) containing bounding boxes in
               format (xmin, ymin, xmax, ymax).
      :target_w: The new image width.
      :target_h: The new image height.

    Returns
      :image: The image with new size.
      :bboxes: The bounding boxes with new size.
    """
    # height, width = tf.unstack(tf.cast(tf.shape(image)[:2], tf.float32))
    height, width = tf.unstack(tf.cast(image_shape, tf.float32))

    # Resize image and keep aspect ratio
    image = tf.image.resize_with_pad(image, target_h, target_w)

    # Re-compute bounding boxes coordinates
    target_h = tf.cast(target_h, tf.float32)
    target_w = tf.cast(target_w, tf.float32)
    ratio = tf.math.minimum(target_w / width, target_h / height)
    resized_height, resized_width = height * ratio, width * ratio
    pad_x = (target_w - resized_width) / 2
    pad_y = (target_h - resized_height) / 2 

    bboxes = bboxes * tf.stack([resized_width, resized_height,
                                resized_width, resized_height]) + \
            tf.stack([pad_x, pad_y, pad_x, pad_y])
    bboxes /= tf.stack([target_w, target_h, target_w, target_h])
    return image, bboxes


@tf.function
def random_brightness(image, max_delta=0.):
    return tf.image.random_brightness(image, max_delta)


@tf.function
def random_hue(image, max_delta):
    return tf.image.random_hue(image, max_delta)
