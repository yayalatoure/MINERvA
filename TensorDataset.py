# -*- coding: utf-8 -*-
# =============================================================================
# Modules
# =============================================================================

# Import required libraries
import h5py as h5
import numpy as np
from tensorflow.python.framework import dtypes

# =============================================================================
# Functions for read and formating dataset
# =============================================================================

def dense_to_one_hot(labels_dense, num_classes):
    # Convert class labels from scalars to one-hot vectors
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def read_file(path):
    # Read H5 file
    f = h5.File(path, "r")
    # Get and print list of datasets within the H5 file
    datasetNames = [n for n in f.keys()]
    for n in datasetNames:
        print(n)
    return f

def formating(file, Tsize, Esize):
    # input: f -> h5.file object
    Ids = file['eventids']
    hits_u = file['hits-u-minerva13Cmc']
    hits_v = file['hits-v-minerva13Cmc']
    hits_x = file['hits-x-minerva13Cmc']
    plane_c = file['planecodes']
    segments = file['segments']
    zs = file['zs']

    # To numpy array
    plane_c = np.array(plane_c)

    # To one hot
    plane_c = dense_to_one_hot(plane_c, 67)

    # Delete extra component
    hits_x = np.squeeze(hits_x, axis=1)
    hits_u = np.squeeze(hits_u, axis=1)
    hits_v = np.squeeze(hits_v, axis=1)

    # Reshape images to vectors
    shape_x = np.shape(hits_x)
    hits_x = np.reshape(hits_x, (shape_x[0], shape_x[1] * shape_x[2]))

    # Shuffle dataset array
    perm = np.arange(Tsize + Esize)
    np.random.shuffle(perm)

    hitsx_shuffled = hits_x[perm]
    planec_shuffled = plane_c[perm]

    # Training and evaluation planes hits_x
    hitsx_train = hitsx_shuffled[0:Tsize]
    hitsx_eval = hitsx_shuffled[Tsize:]

    # Training and evaluation labels
    planec_train = planec_shuffled[0:Tsize]
    planec_eval = planec_shuffled[Tsize:]

    return hitsx_train, planec_train, hitsx_eval, planec_eval

def read_data(path, Tsize, Esize):
    file = read_file(path)
    hitsx_train, planec_train, hitsx_eval, planec_eval = formating(file, Tsize, Esize)

    minerva_train = TensorDataset(hitsx_train, planec_train)
    minerva_eval = TensorDataset(hitsx_eval, planec_eval)

    return minerva_train, minerva_eval

class TensorDataset(object):
    # Inicialización
    def __init__(self,
                 images,
                 labels,
                 one_hot=True,
                 dtype=dtypes.float32,
                 reshape=True,
                 seed=None):

        assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))

        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=False):

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate(
                (images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]
