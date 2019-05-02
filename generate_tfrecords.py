# coding: utf-8


import tensorflow as tf
import numpy as np
import os
import random
import multiprocessing as mp
from PIL import Image


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
    img_data = Image.open(filename)
    img_data = img_data.tostring()
    example = tf.train.Example(
        features=(tf.train.Features(
            feature={
                'name': _bytes_feature(bytes(name, encoding='utf-8')),
                'img_data': _bytes_feature(img_data),
                'label': _int64_feature(label)
            })))
    return example


def write_tfrecords(output, files):
    with tf.python_io.TFRecordWriter(output) as writer:
        for filename, label in files:
            print(filename)
            example = _make_example(filename, label)
            writer.write(example.SerializeToString())


def main():
    total_classes = [os.path.join(FLAGS.data_dir, d) for d in os.listdir(FLAGS.data_dir) if os.path.isdir(d)]
    # total_files ==> [(abs_file1 , class_1),(abs_file2, class_2),...]
    total_files = [(os.path.join(d, f), label) for d, label in enumerate(total_classes) for f in os.listdir(d)]
    random.seed(1234)
    random.shuffle(total_files)
    if FLAGS.train_eval:
        eval_num = FLAGS.eval_percent*len(total_files)
        train_num = len(total_files)-int(eval_num)
        eval_data = total_files[:int(eval_num)]
        train_data = total_files[int(eval_data):]
        train_num_per_tfrecord = train_num//FLAGS.train_tfrecords_num
        eval_num_per_tfrecord = eval_num//FLAGS.eval_tfrecords_num
    else:
        train_data = total_files
        train_num_per_tfrecord = train_data//FLAGS.train_tfrecords_num
    # 多进程写入tfrecords文件
    writer_args = []
    for i in range(0, len(train_data), train_num_per_tfrecord):
        tfrec_name = 'train_'+str(i)+'_of_'+str(FLAGS.train_tfrecords_num)
        writer_args.append([tfrec_name, train_data[i:i+train_num_per_tfrecord]])
    if FLAGS.train_eval:
        for i in range(0, len(eval_data), eval_num_per_tfrecord):
            tfrec_name = 'eval_'+str(i)+'_of_'+str(FLAGS.eval_tfrecords_num)
            writer_args.append([tfrec_name, eval_data[i:i+eval_num_per_tfrecord]])
    pool = mp.Pool(FLAGS.num_processes)
    pool.starmap(write_tfrecords, writer_args)


if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string('data_dir', '', """Directory of data to be converted to tfrecords."""
                        """NOTE that each class should be stored in a child directory.""")
    flags.DEFINE_string('num_processes', '10', """Number of processes to process data.""")
    flags.DEFINE_boolean('train_eval', True, """Dataset split for train and evaluation.""")
    flags.DEFINE_float('eval_percent', 0.2, """Evaluation data ration in total dataset.""")
    flags.DEFINE_integer('train_tfrecords_num', 10, """Number of tfrecords to store train dataset.""")
    flags.DEFINE_integer('eval_tfrecords_num', 1, """Number of tfrecords to store evaluation dataset.""")
    flags.DEFINE_string('outdir', '', """Output directory for tfrecords""")
    FLAGS = flags.FLAGS
    main()
