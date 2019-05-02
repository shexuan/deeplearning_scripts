# coding: utf-8

import tensorflow as tf
import numpy as np
import os
import sys
import time
from datetime import datetime
import parse_tfrecords
from nets import nets_factory

slim = tf.contrib.slim


def inference(inputs, model_name, num_classes=2, weight_decay=0.0, is_training=True):
    '''return logits and predictions
    '''
    assert model_name in nets_factory.networks_map, \
        "Invalid model '{}', which does NOT exist in nets."
    net_fn = nets_factory.get_network_fn(model_name, num_classes, weight_decay, is_training)
    model_image_size = net_fn.default_image_size
    assert model_image_size == FLAGS.image_size, \
        " Inconsistent image size with model default image size."
    logits, endpoints = net_fn(inputs)
    if 'AuxLogits' in endpoints:
        auxiliary_logits = endpoints['AuxLogits']
        return [logits, auxiliary_logits], endpoints['Predictions']
    return [logits], endpoints['predictions']


def loss(logits, labels):
    '''get total loss
    '''
    tf.losses.softmax_cross_entropy(logits[0], labels)
    if len(logits) == 2:
        tf.losses.softmax_cross_entropy(logits[1], labels)
    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
    return total_loss


# def _get_variables_to_restore():
#     '''Returns a list of variables to train.
#     '''
#   if FLAGS.trainable_scopes is None:
#       return tf.trainable_variables()
#   else:
#       scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

#   variables_to_train = []
#   for scope in scopes:
#       variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
#       variables_to_train.extend(variables)
#   return variables_to_train


def init_fn(sess):
    '''
    restore model prameters with fine-tune mode or pretrained mode.
    fine-tune mode means transfer learning, the saved ckpt model trained from other dataset and
    there is a little difference in model structure.
    In pretrained mode, the saved ckpt model trained with the same model structure.
    '''
    if FLAGS.ckpt_path:
        if FLAGS.fine_tune:
            exclusions = []
            if FLAGS.checkpoint_exclude_scopes:
                exclusions = [scope.strip()
                              for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

            # TODO(sguada) variables.filter_variables()
            variables_to_restore = []
            for var in slim.get_model_variables():
                for exclusion in exclusions:
                    if var.op.name.startswith(exclusion):
                        break
                else:
                    variables_to_restore.append(var)
            init_fn = slim.assign_from_checkpoint_fn(FLAGS.ckpt_path,
                                                     variables_to_restore,
                                                     ignore_missing_vars=True)

        elif FLAGS.pretrained:
            init_fn = slim.assign_from_checkpoint_fn(FLAGS.ckpt_path,
                                                     slim.get_model_variables(),
                                                     ignore_missing_vars=True)
        init_fn(sess)


def train(FLAGS):
    is_training = tf.placeholder(tf.bool)
    train_tfrecords = [os.path.join(FLAGS.train_dir, tfrecord) for tfrecord in os.listdir(FLAGS.train_dir)]
    eval_tfrecords = [os.path.join(FLAGS.eval_dir, tfrecord) for tfrecord in os.listdir(FLAGS.eval_dir)]
    training_dataset = parse_tfrecords.preprocess(train_tfrecords,
                                                  batch_size=FLAGS.batch_size,
                                                  is_training=is_training,
                                                  num_threads=4,
                                                  epochs=-1,
                                                  image_size=FLAGS.image_size)
    eval_dataset = parse_tfrecords.preprocess(eval_tfrecords,
                                              batch_size=FLAGS.batch_size,
                                              is_training=is_training,
                                              num_threads=4,
                                              epochs=-1,
                                              image_size=FLAGS.image_size)
    handle = tf.placeholder(tf.string, shape=[])
    # create a feedable iterator
    iterator = tf.data.Iterator.from_string_handle(
        handle, training_dataset.output_types, training_dataset.output_shapes)
    _, img_data, labels = iterator.get_next()
    # different initialize method for different feedable dataset iterator
    training_iterator = training_dataset.make_one_shot_iterator()
    eval_iterator = eval_dataset.make_initializable_iterator()
    logits, predictions = inference(img_data,
                                    model_name=FLAGS.model_name,
                                    num_classes=FLAGS.num_classes,
                                    is_training=is_training)
    slim.losses.softmax_cross_entropy(logits, labels)
    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
    # loss_ema = tf.train.ExponentialMovingAverage(decay=0.9)
    # ema_total_losses = loss_ema.apply([total_losses])
    # with tf.control_dependencies([ema_total_losses]):
    #     total_losses = tf.identity(total_losses)
    glob_steps = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(FLAGS.init_lr,
                                    global_step=glob_steps,
                                    decay_steps=FLAGS.decay_step,
                                    decay_rate=FLAGS.decay_rate,
                                    staircase=False)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):  # 为了使得batch normalization参数能正常更新
        opt = (tf.train.MomentumOptimizer(lr, 0.9)
               .minimize(total_loss, global_step=glob_steps))
    var_ema = tf.train.ExponentialMovingAverage(decay=0.999, num_updates=glob_steps)
    with tf.control_dependencies([opt]):  # 先进行opt梯度传播更新参数，再进行滑动平均更新参数
        train_op = var_ema.apply(tf.trainable_variables()+tf.moving_average_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_handle = sess.run(training_iterator.string_handle())
        eval_handle = sess.run(eval_iterator.string_handle())

        init_fn(sess)
        saver = tf.train.Saver(max_to_keep=100)
        # start training
        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, t_loss = sess.run([train_op, total_loss],
                                 feed_dict={is_training: True, handle: training_handle})
            duration = time.time() - start_time
            if step % 10 == 0:
                        # output the training logs
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (datetime.now(), step, t_loss, examples_per_sec, duration))
                if step % 5000 == 0:
                    # save model
                    ckpt_file = os.path.join(FLAGS.save_dir, FLAGS.save_ckpt_name)
                    saver.save(sess, ckpt_file, global_step=step)
                    # evaluate the model
                    sess.run(eval_iterator.initializer)  # 使用测试集评估时每次评估都重新初始化
                    while True:
                        preds = np.array([])
                        start_time = time.time()
                        try:  # 分batch计算预测结果并进行汇总
                            correct_prediction = tf.cast(
                                tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1)), tf.float32)
                            batch_preds = sess.run(correct_prediction,
                                                   feed_dict={is_training: False, handle: eval_handle})
                            preds = np.append(preds, batch_preds)
                        except tf.errors.OutOfRangeError:
                            duration_preds = time.time() - start_time
                            avg_accuracy = preds.mean()
                            pred_results = ('{0}: step {1}, avg_accuracy = {2:.2f} (Total predictions cost {4} seconds)')
                            print(pred_results.format(datetime.now(), step, avg_accuracy, duration_preds))


if __name__ == "__main__":
    tf.app.flags.DEFINE_integer('image_size', 224, """Input image size.""")
    tf.app.flags.DEFINE_integer('num_classes', 2, """Total classes number of the dataset.""")
    tf.app.flags.DEFINE_integer('batch_size', 64, """Image numbers for each batch.""")
    tf.app.flags.DEFINE_integer('init_lr', 0.01, """Initial learning rate.""")
    tf.app.flags.DEFINE_integer('decay_step', 50000, """Steps after which learning rate decays.""")
    tf.app.flags.DEFINE_integer('decay_rate', 0.5, """Learning dacay factor for per decay.""")
    tf.app.flags.DEFINE_string('model_name', 'resnet_v2_101', """Name of the model which to use.""")
    tf.app.flags.DEFINE_boolean('fine_tune', False, """Transfer learning with pretrained model."""
                                """If True, parameters of the last layers must be droped and reinitialed.""")
    tf.app.flags.DEFINE_boolean('pretrained', False, """Whether used a model pretrained.""")
    tf.app.flags.DEFINE_string('ckpt_path', '', """Pretrained model checkpoint file path."""
                               """Fine-tune and pretrained mode need the checkpoint file.""")
    tf.app.flags.DEFINE_string('train_data', '', """Data directory containing training tfrecords.""")
    tf.app.flags.DEFINE_string('save_dir', '', """Path to save model checkpoint.""")
    tf.app.flags.DEFINE_string('save_ckpt_name', 'resnetV2_101', """Name of the saved checkpoint name.""")
    tf.app.flags.DEFINE_integer('max_steps', 10000001, """Number of batches to run.""")
    tf.app.flags.DEFINE_string('eval_data', '', """Data directory containing evaluation tfrecords""")
    # tf.app.flags.DEFINE_string('optimizer', 'rmsprop', """The name of the optimizer, one of "adadelta","""
    #     """"adagrad", "adam","ftrl", "momentum", "sgd" or "rmsprop".""")
    tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', None, 'Comma-separated list of scopes of variables to exclude when restoring '
                               'from a checkpoint.')

    global FLAGS
    FLAGS = tf.app.flags
