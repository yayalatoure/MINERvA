from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
import matplotlib as mpl
import h5py as h5

tf.logging.set_verbosity(tf.logging.INFO)

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer : [batch_size, image_width, image_height, channels]
    input_layer = tf.reshape(features["x"], [-1, 127, 50, 1])

    # Convolutional Layer N°1
    # padding = same => se considera zero padding
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer N°1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer N°2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer N°2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten pooling layer N°2
    pool2_flat = tf.reshape(pool2, [-1, 31 * 12 * 64])

    # Dense Layer N°1
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=67)


    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
         mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and eval data
    # Read H5 file
    f = h5.File('..\Data\dataset_minerva.hdf5', 'r')
    # Get and print list of datasets within the H5 file
    datasetNames = [n for n in f.keys()]
    print('\nFields of hdf5 data file: \n')
    for n in datasetNames:
        print(n)

    Ids = f['eventids']
    hits_u = f['hits-u-minerva13Cmc']
    hits_v = f['hits-v-minerva13Cmc']
    hits_x = f['hits-x-minerva13Cmc']
    plane_c = f['planecodes']
    segments = f['segments']
    zs = f['zs']

    # Formating Data
    Tsize = 40000
    Esize = 10000
    training_hu = np.squeeze(hits_u[0:Tsize], axis=1)
    training_hv = np.squeeze(hits_v[0:Tsize], axis=1)
    training_hx = np.squeeze(hits_x[0:Tsize], axis=1)

    eval_hu = np.squeeze(hits_u[Tsize:], axis=1)
    eval_hv = np.squeeze(hits_v[Tsize:], axis=1)
    eval_hx = np.squeeze(hits_x[Tsize:], axis=1)

    # Training and Eval Data
    train_huInput = np.reshape(training_hu, (Tsize, 127 * 25))
    train_hvInput = np.reshape(training_hv, (Tsize, 127 * 25))
    train_data = np.reshape(training_hx, (Tsize, 127 * 50))
    eval_hxInput = np.reshape(eval_hu, (Esize, 127 * 25))
    eval_hvInput = np.reshape(eval_hv, (Esize, 127 * 25))
    eval_data = np.reshape(eval_hx, (Esize, 127 * 50))

    # Training and Eval Labels
    # Output Layer solo recibe enteros de 32 o 64
    train_labels = plane_c[0:Tsize]
    train_labels = train_labels.astype(np.int32, copy=False)
    eval_labels = plane_c[Tsize:]
    eval_labels = eval_labels.astype(np.int32, copy=False)

    print('\nSize of training data is: ', np.shape(train_data))
    print('Size of training labels is: ', np.shape(train_labels))
    print('Size of evaluation data is: ', np.shape(eval_data))
    print('Size of evaluation data is: ', np.shape(eval_labels))

    image = np.reshape(train_data[1000], (127, 50))
    print('Size of image is: \n', np.shape(image))
    #show(image)

    # Train Network
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="C:/Users/lalo/Desktop/CCTVal/MINERvA/VortexDetection/Checkpoints")
    # Directorio anterior para los checkpoints de datos del modelo serán guardados.

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=1000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    # La evaluacion se raliza con la m+etrica definida
    # anteriormente en: cnn_model_fn -> eval_metric_ops
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()

















