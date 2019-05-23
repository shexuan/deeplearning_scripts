# coding: utf-8

from __future__ import division
import tensorflow as tf
import numpy as np
import random
import os
from PIL import Image


def check_dtype(img_data, dtype=tf.float32):
    '''检查数据类型是否为tf.float32
    '''
    if img_data.dtype != tf.float32:
        img_data = tf.image.convert_image_dtype(img_data, tf.float32)
    return img_data


def resize_image(name, img_data, label, image_size):
    '''调整图像大小
    '''
    # 在tensorflow 1.13(或者1.12？)版本前tf.reshape不支持shape参数为多个tensor，
    # 故无法对tfrecord解析出来的img_data进行reshape到原来的shape，只能同意reshape到某一个shape
    #input_size = img_data.get_shape().as_list()[1]
    # if image_size != input_size:
    img_data = tf.image.resize_images(img_data, [image_size, image_size], method=np.random.randint(4))
    img_data = tf.reshape(img_data, shape=[image_size, image_size, 3])   # 这一步仅仅是为了使得img_data的shape从[new_width, new_height, ?]变成[new_width, new_height, 3]
    return name, img_data, label


def distort_color(name, img_data, label):
    '''随机调整图像的色彩，包括对比度，亮度，色彩，饱和度和色相等
    不同的调整顺序得到的结果不同
    '''
    def distort_brightness(img_data, max_delta=32. / 255.):
        return tf.image.random_brightness(img_data, max_delta=max_delta)

    def distort_saturation(img_data, lower=0.5, upper=1.5):
        return tf.image.random_saturation(img_data, lower=lower, upper=upper)

    def distort_hue(img_data, max_delta=0.2):
        return tf.image.random_hue(img_data, max_delta=max_delta)

    def distort_contrast(img_data, lower=0.5, upper=1.5):
        return tf.image.random_contrast(img_data, lower=lower, upper=upper)

    func_list = [distort_brightness, distort_contrast, distort_hue, distort_saturation]
    random.shuffle(func_list)
    for func in func_list:
        img_data = func(img_data)
    img_data_adjusted = tf.clip_by_value(img_data, 0.0, 1.0)
    return name, img_data_adjusted, label


def crop_image(name, img_data, label, bbox=None):
    '''裁剪某个bounding boxes中的图像
    '''
    # 若未提供bounding boxes, 则从整个图像中裁剪
    if not bbox:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=(1, 1, 4))
    # 随机截取图像，减小需要关注的物体大小对图像识别算法的影响
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(img_data), bounding_boxes=bbox)
    distort_img = tf.slice(img_data, bbox_begin, bbox_size)
    # 将随机截取的图像调整为神经网络输入层的大小，大小调整的算法随机选择
    return name, distort_img, label


def flip_image(name, img_data, label):
    '''随机上下左右或不翻转图像
    '''
    # 随机左右翻转图像
    if np.random.randint(2):
        img_data = tf.image.random_flip_left_right(img_data)
    if np.random.randint(2):
        img_data = tf.image.random_flip_up_down(img_data)
    return name, img_data, label


def _parse_tfrecord(example_proto):
    features = {
        'name': tf.FixedLenFeature((), tf.string),
        'img_data': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.int64),
        'height': tf.FixedLenFeature((), tf.int64),
        'width': tf.FixedLenFeature((), tf.int64)
    }
    parsed_features = tf.parse_single_example(example_proto, features=features)
    img_data = tf.image.decode_image(parsed_features['img_data'], channels=3)
    shape = tf.reshape([parsed_features['height'], parsed_features['width'], 3], [-1])
    img_data = tf.reshape(img_data, shape=shape)
    img_data = tf.image.convert_image_dtype(img_data, tf.float32)  # 归一化，将像素值转化为[0,1]之间
    name = parsed_features['name']
    label = parsed_features['label']
    return name, img_data, label


def preprocess(tfrecords, batch_size=32, is_training=True, num_threads=4, epochs=-1, image_size=244):
    with tf.device('/cpu:0'):
        dataset = tf.data.TFRecordDataset(tfrecords)
        dataset = dataset.map(_parse_tfrecord)
        if is_training:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=batch_size*100, count=epochs))
            #tf.data.experimental.shuffle_and_repeat(buffer_size=10000, count=epochs)
            dataset = (dataset
                       #.map(flip_image, num_parallel_calls=num_threads)
                       #.map(distort_color, num_parallel_calls=num_threads)
                       #.map(crop_image, num_parallel_calls=num_threads)
                       .map(lambda *x: resize_image(*x, image_size=image_size),
                            num_parallel_calls=num_threads))
            dataset = dataset.batch(batch_size)  # tf 1.10以上才添加的参数
            #dataset = dataset.batch_and_drop_remainder(batch_size)
            dataset = dataset.prefetch(1)  # 取出一个batch
        else:
            dataset = dataset.map(lambda *x: resize_image(*x, image_size=image_size),
                                  num_parallel_calls=num_threads)
            dataset = dataset.batch(batch_size)  # 这时候不能使用drop_remainder=True，因为预测用到了所有图片
            dataset = dataset.prefetch(batch_size)
        #iterator = dataset.make_one_shot_iterator()
        # iterator = dataset.make_initializable_iterator()
        return dataset


if __name__ == '__main__':
    tfrecords = [f for f in os.listdir('.') if f.endswith('tfrecords')]
    preprocess(tfrecords)
