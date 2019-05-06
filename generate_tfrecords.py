# coding: utf-8


import tensorflow as tf
import numpy as np
import os
import math
import random
import multiprocessing as mp
from PIL import Image
from collections import defaultdict


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    # 注意：这里的value必须以value=value的形式放进去，而不能直接放个value进去，
    # 不然会报错


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _make_example(filename, label):
    name = filename.split('/')[-1]
    img_data = tf.gfile.FastGFile(filename, 'rb').read()  # 这种方式更加节省内存
    img = Image.open(filename)
    width = img.width
    height = img.height
    example = tf.train.Example(
        features=(tf.train.Features(
            feature={
                'name': _bytes_feature(bytes(name, encoding='utf-8')),
                'img_data': _bytes_feature(img_data),
                'label': _int64_feature(label),
                'height': _int64_feature(height),
                'width': _int64_feature(width)
            })))
    return example


def write_tfrecords(output, files):
    with tf.python_io.TFRecordWriter(output) as writer:
        for filename, label in files:
            print(filename)
            example = _make_example(filename, label)
            writer.write(example.SerializeToString())


def main():
    print(os.listdir(FLAGS.data_dir))
    total_classes = [os.path.join(FLAGS.data_dir, d) for d in os.listdir(FLAGS.data_dir)
                     if os.path.isdir(os.path.join(FLAGS.data_dir, d))]
    print(total_classes)
    train_data = []
    eval_data = []
    for label, dir_ in enumerate(total_classes):
        data = [(os.path.join(dir_, f), label) for f in os.listdir(dir_)]
        if FLAGS.train_eval:
            eval_num = FLAGS.eval_percent * len(data)
            eval_data.extend(data[:int(eval_num)])
            train_data.extend(data[int(eval_num):])
    train_num = len(train_data)
    eval_num = len(eval_data)
    print(train_num, FLAGS.train_tfrecords_num)
    train_num_per_tfrecord = math.ceil(train_num / FLAGS.train_tfrecords_num)
    eval_num_per_tfrecord = math.ceil(eval_num / FLAGS.eval_tfrecords_num)

    random.seed(1234)
    random.shuffle(train_data)
    random.seed(1234)
    random.shuffle(eval_data)

    # 多进程写入tfrecords文件
    writer_args = []
    n = 1
    for i in range(0, len(train_data), train_num_per_tfrecord):
        tfrec_name = FLAGS.outdir + '/train_' + str(n) + '_of_' + str(FLAGS.train_tfrecords_num)
        writer_args.append([tfrec_name, train_data[i:i + train_num_per_tfrecord]])
        n += 1
    if FLAGS.train_eval:
        n = 1
        for i in range(0, len(eval_data), eval_num_per_tfrecord):
            tfrec_name = FLAGS.outdir + '/eval_' + str(n) + '_of_' + str(FLAGS.eval_tfrecords_num)
            writer_args.append([tfrec_name, eval_data[i:i + eval_num_per_tfrecord]])
            n += 1
    pool = mp.Pool(FLAGS.num_processes)
    pool.starmap(write_tfrecords, writer_args)


if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string('data_dir', '', """Directory of data to be converted to tfrecords."""
                        """NOTE that each class should be stored in a child directory.""")
    flags.DEFINE_integer('num_processes', 10, """Number of processes to process data.""")
    flags.DEFINE_boolean('train_eval', True, """Dataset split for train and evaluation.""")
    flags.DEFINE_float('eval_percent', 0.2, """Evaluation data ration in total dataset.""")
    flags.DEFINE_integer('train_tfrecords_num', 10, """Number of tfrecords to store train dataset.""")
    flags.DEFINE_integer('eval_tfrecords_num', 1, """Number of tfrecords to store evaluation dataset.""")
    flags.DEFINE_string('outdir', '', """Output directory for tfrecords""")

    global FLAGS
    FLAGS = flags.FLAGS

    main()
