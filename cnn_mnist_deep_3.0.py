""" import libraries """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os
import itertools
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import confusion_matrix

import numpy as np
import tensorflow as tf
from matplotlib import pyplot
import matplotlib as plt


FLAGS = None
LOGDIR = 'C:/Users/lalo/Desktop/CCTVal/MINERvA/checkpoints3.0/'

def show_image(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=plt.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',
                          cmap=pyplot.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    pyplot.imshow(cm, interpolation='nearest', cmap=cmap)
    pyplot.title(title)
    pyplot.colorbar()
    tick_marks = np.arange(len(classes))
    pyplot.xticks(tick_marks, classes, rotation=45)
    pyplot.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pyplot.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    pyplot.tight_layout()
    pyplot.ylabel('True label')
    pyplot.xlabel('Predicted label')

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

def deepnn(x):
    """
    deepnn builds the graph for a deep net for classifying digits.
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
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    conv_1 = conv_layer(x_image, 5, 5, 1, 32, "conv_1")

    pool_1 = pool_layer(conv_1, 2, 2, "pool_1")

    conv_2 = conv_layer(pool_1, 5, 5, 32, 64, "conv_2")

    pool_2 = pool_layer(conv_2, 2, 2, "pool_2")

    full_1 = fc_layer(pool_2, 7*7*64, 1024, True, "full_1")

    drop_1, keep_prob = dropout(full_1)

    full_2 = fc_layer(drop_1, 1024, 10, False, "full_2")

    return full_2, keep_prob

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
        tf.summary.scalar("xent", xent)
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
        actual_pred = tf.argmax(y_conv, 1)
        correct_pred = tf.equal(actual_pred, tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
        return accuracy, correct_pred, actual_pred


def main(_):

    tf.reset_default_graph()
    sess = tf.Session()

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)

    xent = cross_entropy(y_, y_conv, "xent")

    train_step = training(xent, "adam_optimizer")

    accuracy, correct_pred, actual_pred = accuracy_measure(y_, y_conv, "accuracy")

    """ Logging configuration """
    summ = tf.summary.merge_all()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    # filewriter is how we write the summary protocol buffers to disk
    train_writer = tf.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)
    ## Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(train_writer, config)


    """ Training Neural Network """
    sess.run(tf.global_variables_initializer())
    for i in range(7000):
        batch = mnist.train.next_batch(100)
        if i % 100 == 0:
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

        train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.8})


    test_data = mnist.test.images
    test_labels = mnist.test.labels
    '''Evaluation - Get test accuracy'''
    batch_correct_cts = []
    batch_predict_cts = []
    count = 0
    for batch_data, batch_labels in batches(test_data, test_labels, 100):
        count += 1
        ct_correct, preds = sess.run([correct_pred, actual_pred], feed_dict={
                            x: batch_data, y_: batch_labels, keep_prob: 0.8})
        batch_correct_cts.append(ct_correct.sum())
        batch_predict_cts.append(preds)

   #last_batch_labels = np.argmax(test_labels[len(test_labels)-100:], axis=1)
   #
   #print(preds)
   #print(last_batch_labels)
   #print('\nTest accuracy: ')
   #print(sum(batch_correct_cts) / len(test_labels))

   #print('\nultimo batch accuracy: ')#
   #print(sum(np.int32(np.equal(preds, last_batch_labels))) / 100)


    print('\n Confusion Matrix')
    predictions = np.concatenate(batch_predict_cts, axis=0)
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(predictions, np.argmax(test_labels, axis=1))

    np.set_printoptions(precision=2)
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # Plot non-normalized confusion matrix
    pyplot.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    # Plot normalized confusion matrix
    pyplot.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    pyplot.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

