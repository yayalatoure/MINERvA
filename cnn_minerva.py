
""" import libraries """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot
import matplotlib as mpl

from TensorDataset import *


FLAGS = None
LOGDIR = 'C:/Users/lalo/Desktop/CCTVal/MINERvA/checkpoints/'
PATH_DATA = 'C:/Users/lalo/Google Drive/CCTVal/MINERvA/VortexDetection/Data/dataset_minerva.hdf5'

TRAIN_SIZE = 49000
EVAL_SIZE = 1000

HX_SIZE_VECTOR = 127*50
HUV_SIZE_VECTOR = 127*25
CATEGORIES = 67


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
        '''Logging Histograms of Conv Layer'''
        # tf.summary.histogram("weights", w)
        # tf.summary.histogram("biases", b)
        # tf.summary.histogram("activations", act)
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
        # tf.summary.histogram("weights", w)
        # tf.summary.histogram("biases", b)
        # tf.summary.histogram("activations", act)
        return act

def dropout(input, name="dropout"):
    """
    :param input: fully connected input layer
    :param keep_prob: probability de dropout
    :return:
    """
    with tf.name_scope(name):
        keep_prob = tf.placeholder(tf.float32)
        drop = tf.nn.dropout(input, keep_prob)
        return drop, keep_prob

def deepnn_hitsX(x):
    """
    deepnn builds the graph for a deep neural network for classifying MINERvA data.
    :param x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
    :return: A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 127, 50, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    conv_1 = conv_layer(x_image, 5, 5, 1, 32, "conv_1")

    pool_1 = pool_layer(conv_1, 2, 2, "pool_1")

    conv_2 = conv_layer(pool_1, 5, 5, 32, 64, "conv_2")

    pool_2 = pool_layer(conv_2, 2, 2, "pool_2")

    full_1 = fc_layer(pool_2, 32*13*64, 1024, True, "full_1")

    drop_1, keep_prob = dropout(full_1)
    #
    full_2 = fc_layer(drop_1, 1024, CATEGORIES, False, "full_2")

    return full_2, keep_prob

def deepnn_hitsUV(u, v):
    """
    deepnn builds the graph for a deep neural network for classifying MINERvA data.
    :param x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
    :return: A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        u_image = tf.reshape(u, [-1, 127, 25, 1])
        v_image = tf.reshape(v, [-1, 127, 25, 1])

    # Convolutional layer for U plane
    conv_1u = conv_layer(u_image, 8, 3, 1, 32, "conv_1")
    pool_1u = pool_layer(conv_1u, 2, 2, "pool_1")
    conv_2u = conv_layer(pool_1u, 8, 3, 32, 64, "conv_2")
    pool_2u = pool_layer(conv_2u, 2, 2, "pool_2")
    full_1u = fc_layer(pool_2u, 32*7*64, 1024, True, "full_1")
    drop_1u, keep_prob = dropout(full_1u)
    full_2u = fc_layer(drop_1u, 1024, CATEGORIES, False, "full_2")

    # Convolutional layer for V plane
    conv_1v = conv_layer(v_image, 8, 3, 1, 32, "conv_1")
    pool_1v = pool_layer(conv_1v, 2, 2, "pool_1")
    conv_2v = conv_layer(pool_1v, 8, 3, 32, 64, "conv_2")
    pool_2v = pool_layer(conv_2v, 2, 2, "pool_2")
    full_1v = fc_layer(pool_2v, 32*7*64, 1024, True, "full_1")
    drop_1v, keep_prob = dropout(full_1v)
    full_2v = fc_layer(drop_1v, 1024, CATEGORIES, False, "full_2")

    return full_2u, full_2v, keep_prob

def cross_entropy(y_, y_conv, name="xent"):
    """
    :param y_: label from dataset
    :param y_conv: label predicted by cnn
    :param name: loss function name
    :return: cross entropy xent object
    """
    with tf.name_scope(name):
        xent = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
        xent = tf.reduce_mean(xent)
        # tf.summary.scalar("xent", xent)
        return xent

def training(xent, name="adam_optimizer"):
    """
    :param xent: cross entropy metric to opotmize
    :param name: optimizer's name
    :return: train step
    """
    with tf.name_scope(name):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(xent)
        return train_step

def accuracy_measure(y_, y_conv, name="accuracy"):
    """
    :param y_: label from dataset
    :param y_conv: label predicted by cnn
    :param name: accuracy measure name
    :return: accuracy object
    """
    with tf.name_scope(name):
        prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        # tf.summary.scalar("accuracy", accuracy)
        return accuracy, prediction, correct_prediction

def main(_):

    tf.reset_default_graph()
    sess = tf.Session()

    # Import MINERvA data
    minerva_train, minerva_test = read_data(PATH_DATA, TRAIN_SIZE, EVAL_SIZE)
    print('\n type mnist datasets: ', type(minerva_train))

    # Create the model
    x = tf.placeholder(tf.float32, [None, HX_SIZE_VECTOR])
    u = tf.placeholder(tf.float32, [None, HUV_SIZE_VECTOR])
    v = tf.placeholder(tf.float32, [None, HUV_SIZE_VECTOR])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, CATEGORIES])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn_hitsX(x)

    xent = cross_entropy(y_, y_conv, "xent")

    train_step = training(xent, "adam_optimizer")

    accuracy, correct_prediction = accuracy_measure(y_, y_conv, "accuracy")

    """ Logging configuration """
    # summ = tf.summary.merge_all()
    saver = tf.train.Saver()
    # filewriter is how we write the summary protocol buffers to disk
    train_writer = tf.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)
    ## Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(train_writer, config)


    """ Training Neural Network """
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        batch = minerva_train.next_batch(10)
        if i % 10 == 0:
            print('iteración %g: ' % i)
            """Evaluando accuracy y xent"""
            train_accuracy = accuracy.eval(session=sess, feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 0.8})

            train_loss = xent.eval(session=sess, feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 0.8})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            print('step %d, training loss %g' % (i, train_loss))

            with sess.as_default():
                acc = tf.Summary(value=[tf.Summary.Value(tag='train_accuracy', simple_value=train_accuracy)])
                loss = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss)])
                train_writer.add_summary(acc, i)
                train_writer.add_summary(loss, i)

        # if i % 100 == 0:
        #     test_accuracy = accuracy.eval(session=sess, feed_dict={
        #                     x: minerva_test.images, y_: minerva_test.labels, keep_prob: 0.8})
        #     print('\ntest accuracy: %g' % test_accuracy)
        #
        #     # [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={
        #     #     x: batch[0], y_: batch[1], keep_prob: 1.0})
        #     # train_writer.add_summary(s, i)
        #     print("train accuracy: ", train_accuracy)
        #     saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)

        train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.8})

    pred_labels = accuracy.run(session=sess, feed_dict={
                 x: minerva_test.images, y_: minerva_test.labels, keep_prob: 0.8})


    # test_accuracy = accuracy.eval(session=sess, feed_dict={
    #                 x: minerva_test.images, y_: minerva_test.labels, keep_prob: 0.8})
    # print('\ntest accuracy: %g' % test_accuracy)
    # test_cross = correct_prediction.eval(session=sess, feed_dict={
    #              x: minerva_test.images, y_: minerva_test.labels, keep_prob: 0.8})
    # print('\nshape of cross entropy vector: %g' % np.shape(test_cross))
    # print('accuracy form cross_entropy: %g' % (np.sum(test_cross) / np.shape(test_cross)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)



