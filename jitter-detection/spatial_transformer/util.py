"""
util.py
=======

Implementation of utility functions.
"""

import tensorflow as tf

def gen_sampling_grid(height: int, width: int, theta: tf.Tensor):
    """
    Generate a sampling grid for bilinear sampler.

    Params
    ------
    height: int
        Height of the grid.

    width: int
        Width of the grid.

    theta: tf.Tensor
        Affine transformation matrices of shape (batch_count, 2, 3).

    Return Type
    -----------
    tf.Tensor of shape (batch_count, 2, height, width). The second dimension
    has two components (x, y) which are sampling points of the original image
    for each point in the target image.
    """
    batch_count = theta.shape[0]
    x_t, y_t = tf.meshgrid(tf.linspace(-1.0, 1.0, width),
                           tf.linspace(-1.0, 1.0, height))
    flattened_x_t = tf.reshape(x_t, [-1])
    flattened_y_t = tf.reshape(y_t, [-1])
    ones = tf.ones_like(flattened_x_t)
    sampling_grid = tf.stack((flattened_x_t, flattened_y_t, ones))
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack((batch_count, 1, 1)))

    theta = tf.cast(theta, tf.float32)
    sampling_grid = tf.cast(sampling_grid, tf.float32)
    batch_grid = tf.matmul(theta, sampling_grid)
    batch_grid = tf.reshape(batch_grid, (batch_count, 2, height, width))
    return batch_grid

def get_pixel_value(image: tf.Tensor, x: tf.Tensor, y: tf.Tensor):
    """
    Get pixel value for x and y, from 4D tensor.

    Params
    ------
    image: tf.Tensor
        Batch of images in shape (batch_count, height, width, channels).

    x: tf.Tensor
        Tensor of shape (batch_count, height, width,).

    y: tf.Tensor
        Tensor with same shape as x.

    Return Type
    -----------
    tf.Tensor of shape (batch_count, height, width, channels)
    """
    batch_count = x.shape[0]
    height = x.shape[1]
    width = x.shape[2]
    batch_index = tf.reshape(tf.range(0, batch_count), (batch_count, 1, 1))
    b = tf.tile(batch_index, (1, height, width))
    indices = tf.stack((b, y, x), 3)
    return tf.gather_nd(image, indices)

def bilinear_sampler(image: tf.Tensor, x: tf.Tensor, y: tf.Tensor):
    """
    Perform bilinear sampling of the input images according to the normalized
    coordinates.

    Params
    ------
    image: tf.Tensor
        Batch of images in shape (batch_count, height, width, channels).

    x: tf.Tensor
        Normalized x coordinates. Result of gen_sampling_grid.

    y: tf.Tensor
        Normalized y coordinates.

    Return Type
    -----------
    tf.Tensor which is interpolated images.
    """
    height = image.shape[1]
    width = image.shape[2]
    max_y = tf.cast(height - 1, tf.int32)
    max_x = tf.cast(width - 1, tf.int32)
    zero = tf.zeros([], tf.int32)

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    x = 0.5 * ((x + 1.0) * tf.cast(max_x - 1, tf.float32))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y - 1, tf.float32))

    x0 = tf.cast(tf.floor(x), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), tf.int32)
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    I1 = get_pixel_value(image, x0, y0)
    I2 = get_pixel_value(image, x0, y1)
    I3 = get_pixel_value(image, x1, y0)
    I4 = get_pixel_value(image, x1, y1)

    x0 = tf.cast(x0, tf.float32)
    x1 = tf.cast(x1, tf.float32)
    y0 = tf.cast(y0, tf.float32)
    y1 = tf.cast(y1, tf.float32)

    w1 = tf.expand_dims((x1 - x) * (y1 - y), axis=3)
    w2 = tf.expand_dims((x1 - x) * (y - y0), axis=3)
    w3 = tf.expand_dims((x - x0) * (y1 - y), axis=3)
    w4 = tf.expand_dims((x - x0) * (y - y0), axis=3)

    return tf.add_n((w1 * I1, w2 * I2, w3 * I3, w4 * I4))
