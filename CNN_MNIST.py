""" import libraries """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os

from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import tensorflow as tf
from matplotlib import pyplot
import matplotlib as mpl

FLAGS = None
LOGDIR = 'C:/Users/lalo/Desktop/CCTVal/MINERvA/checkpoints3/'

def shuffle(*args):
    "Shuffles list of NumPy arrays in unison"
    state = np.random.get_state()
    for array in args:
        np.random.set_state(state)
        np.random.shuffle(array)

def grouper(iter_, n):
    """Collect data into fixed-length chunks or blocks
     grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
     from python itertools docs"""
    args = [iter(iter_)] * n
    return zip(*args)

def batches(data, labels, batch_size, randomize=True):
    if len(data) != len(labels):
        raise ValueError('Image data and label data must be same size')
    if batch_size > len(data):
        raise ValueError('Batch size cannot be larger than size of datasets')
    if randomize:
        shuffle(data, labels)
    for res in zip(grouper(data, batch_size),
                   grouper(labels, batch_size)):
        yield res

def show_image(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

def conv2d(x, w):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv_layer(input, cr_h, cr_v, depth_in, depth_out, name="conv"):
    """
    :param input: image or volume input to the conv layer
    :param cr_h: receptive field size horizontal
    :param cr_v: receptive field size vertical
    :param depth_in: input volume depth
    :param depth_out: output volume depth
    :param name: name of conv layer
    :return: output volume same size of input image with depth depth_out
    """
    with tf.name_scope(name):
        w = weight_variable([cr_h, cr_v, depth_in, depth_out])
        b = bias_variable([depth_out])
        act = tf.nn.relu(conv2d(input, w) + b)
        return act

def pool_layer(input, ds_h, ds_v, name="pool"):
    """
    :param input: input volume
    :param ds_h: horizontal downsample
    :param ds_v: vertical downsample
    :param name: name of pooling layer
    :return: output volume, same depth donwsampled length and width
    """
    with tf.name_scope(name):
        return tf.nn.max_pool(input, ksize=[1, ds_h, ds_v, 1],
                        strides=[1, ds_h, ds_v, 1], padding='SAME')

def fc_layer(input, flat_size, dense_neurons, flag, name="fc"):
    """
    :param input: input volume
    :param depth_in: input volume depth
    :param depth_out: output volume depth
    :param name: name of fully connected layer
    :return:
    """
    with tf.name_scope(name):
        w = weight_variable([flat_size, dense_neurons])
        b = bias_variable([dense_neurons])
        if flag == True:
            input_flat = tf.reshape(input, [-1, flat_size])
        else:
            input_flat = input
        act = tf.nn.relu(tf.matmul(input_flat, w) + b)
        return act

def dropout(input, keep_prob, name="dropout"):
    """
    :param input: fully connected input layer
    :param keep_prob: probability de dropout
    :return:
    """
    with tf.name_scope(name):
        drop = tf.nn.dropout(input, keep_prob)
        return drop

def fully_connected_layer(incoming_layer, num_nodes, activation_fn=tf.nn.sigmoid, w_stddev=0.5,
                          b_val=0.0, keep_prob=None, name=None):
    incoming_layer = tf.convert_to_tensor(incoming_layer)
    prev_num_nodes = incoming_layer.shape.dims[-1].value

    with tf.name_scope(name, 'fully_connected'):
        tn = tf.truncated_normal([prev_num_nodes, num_nodes],
                                 stddev=w_stddev)
        W = tf.Variable(tn, name='W')
        const = tf.constant(b_val, shape=[num_nodes])
        b = tf.Variable(const, name='bias')

        z = tf.matmul(incoming_layer, W) + b

        a = activation_fn(z) if activation_fn is not None else z
        final_a = a if keep_prob is None else tf.nn.dropout(a, keep_prob)

        return final_a


class MNIST_Model(object):
    def __init__(s):  # non-standard, for abbreviation
        graph = tf.Graph()
        with graph.as_default():
            with tf.name_scope('inputs'):
                s.x = tf.placeholder(tf.uint8, shape=[None, 28*28], name="x")
                s.y = tf.placeholder(tf.float32, shape=[None, 10])

            with tf.name_scope('hyperparams'):
                s.learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')
                s.x_keep_prob = tf.placeholder(tf.float32, [], 'x_keep_prob')
                s.h_keep_prob = tf.placeholder(tf.float32, [], 'h_keep_prob')

            with tf.name_scope('preprocess'):
                s.x_float = tf.cast(s.x, tf.float32)
                #s.x_image = tf.reshape(s.x_float, [-1, 28, 28, 1])
                s.x_dropped = tf.nn.dropout(s.x_float, s.x_keep_prob)
                #s.one_hot_labels = tf.one_hot(s.y, 10)
                #Con el dataset de antes el one_hot ya viene hecho
                s.one_hot_labels = s.y

            # with tf.name_scope('model'):
            #     # First convolutional layer - maps one grayscale image to 32 feature maps.
            #     conv_1 = conv_layer(s.x_image, 5, 5, 1, 32, "conv_1")
            #     pool_1 = pool_layer(conv_1, 2, 2, "pool_1")
            #     conv_2 = conv_layer(pool_1, 5, 5, 32, 64, "conv_2")
            #     pool_2 = pool_layer(conv_2, 2, 2, "pool_2")
            #     full_1 = fc_layer(pool_2, 7 * 7 * 64, 1024, True, "full_1")
            #     drop_1 = dropout(full_1, s.x_keep_prob)
            #     s.out = fc_layer(drop_1, 1024, 10, False, "full_2")

            with tf.name_scope('model'):
                make_fc = fully_connected_layer  # abbreviation

                s.h1 = make_fc(s.x_dropped, 1200, keep_prob=s.h_keep_prob, name='h1')
                s.h2 = make_fc(s.h1, 1200, keep_prob=s.h_keep_prob, name='h2')
                s.h3 = make_fc(s.h2, 1200, keep_prob=s.h_keep_prob, name='h3')
                s.out = make_fc(s.h3, 10, name='out')

            with tf.name_scope('loss'):
                smce = tf.nn.softmax_cross_entropy_with_logits
                s.loss = tf.reduce_mean(smce(logits=s.out, labels=s.one_hot_labels))

            with tf.name_scope('train'):
                opt = tf.train.AdamOptimizer(s.learning_rate)
                s.train = opt.minimize(s.loss)

            with tf.name_scope('global_step'):
                global_step = tf.Variable(0, trainable=False, name='global_step')
                s.inc_step = tf.assign_add(global_step, 1, name='inc_step')

            with tf.name_scope('prediction'):
                s.softmax = tf.nn.softmax(s.out, name="softmax")
                s.prediction = tf.cast(tf.arg_max(s.softmax, 1), tf.int32)
                s.pred_correct = tf.equal(s.y, tf.one_hot(s.prediction, 10))
                s.pred_accuracy = tf.reduce_mean(tf.cast(s.pred_correct, tf.float32))

            s.init = tf.global_variables_initializer()

        s.session = tf.Session(graph=graph)
        s.session.run(s.init)

    def fit(s, train_dict):
        tr_loss, step, tr_acc, _ = s.session.run([s.loss, s.inc_step, s.pred_accuracy,
                                                  s.train], feed_dict=train_dict)
        return tr_loss, step, tr_acc

    def predict(s, test_dict):
        ct_correct, preds = s.session.run([s.pred_correct, s.prediction],
                                          feed_dict=test_dict)
        return ct_correct, preds


def main(_):

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    train_data = mnist.train.images
    train_labels = mnist.train.labels
    shuffle(train_data, train_labels)


    mm = MNIST_Model()
    for epoch in range(10):
        for batch_data, batch_labels in batches(train_data, train_labels, 200, True):
            train_dict = {mm.x: batch_data,
                          mm.y: batch_labels,
                          mm.learning_rate: 0.001,
                          mm.x_keep_prob: 0.8,
                          mm.h_keep_prob: 0.5}
            tr_loss, step, tr_acc = mm.fit(train_dict)
            info_update = "Epoch: {:2d} Step: {:5d} Loss: {:8.2f} Acc: {:5.2f}"
            print(info_update.format(epoch, step, tr_loss, tr_acc))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


















