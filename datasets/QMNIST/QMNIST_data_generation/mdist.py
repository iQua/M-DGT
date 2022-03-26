#!/usr/bin/env python
# -*- coding: utf-8 -*-



''' python inherent libs '''


''' third parts libs '''
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
import numpy as np

''' local custom libs '''

DATA_SIZE = 50000


class DataSet(object):

    def __init__(self,
                 images,
                 questions,
                 labels,
                 dtype=dtypes.float32,
                 reshape=False):
        """Construct a DataSet."""

        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)

        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        if reshape:
            assert images.shape[1] == 3
            images = images.reshape(images.shape[0],
                                    images.shape[2] * images.shape[3] * images.shape[1])
        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)

        self._images = images
        self._questions = questions
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def questions(self):
        return self._questions

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._questions = self.questions[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            questions_rest_part = self._questions[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._questions = self.questions[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            questions_new_part = self._questions[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0),\
                np.concatenate((questions_rest_part, questions_new_part), axis=0),\
                np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._questions[start:end], self._labels[start:end]

# the data dir should be: FLAGS.dataset_dir + "/mdist.npz"


def dense_to_one_hot(labels_dense, num_classes=5):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def read_data_sets(data_dir,
                   one_hot=True,
                   dtype=dtypes.float32,
                   reshape=False):
    data = np.load(data_dir + "/mdist.npz")

    imgs = data['data']
    imgs = np.transpose(imgs, axes=[0, 2, 3, 1])
    print(np.shape(imgs))

    refs = data['refs']
    qs = data['questions'].astype('int32')
    ys = data['labels'].astype('int32')

    if one_hot:
        ys = dense_to_one_hot(ys)

    xref_train, xref_valid, xref_test = refs[
        :DATA_SIZE - 20000], refs[DATA_SIZE - 20000:DATA_SIZE - 10000], refs[DATA_SIZE - 10000:DATA_SIZE]
    xq_train, xq_valid, xq_test = qs[:DATA_SIZE - 20000], qs[DATA_SIZE -
                                                             20000:DATA_SIZE - 10000], qs[DATA_SIZE - 10000:DATA_SIZE]
    train_labels, vaild_labels, test_labels = ys[:DATA_SIZE - 20000].astype('int32'), ys[
        DATA_SIZE - 20000:DATA_SIZE - 10000].astype('int32'), ys[DATA_SIZE - 10000:DATA_SIZE].astype('int32')

    test_scale = data['scales'][DATA_SIZE - 10000:DATA_SIZE]

    imgs_train = imgs[xref_train]
    imgs_valid = imgs[xref_valid]
    imgs_test = imgs[xref_test]

    train = DataSet(imgs_train, xq_train, train_labels, dtype=dtype, reshape=reshape)
    validation = DataSet(imgs_valid,
                         xq_valid,
                         vaild_labels,
                         dtype=dtype,
                         reshape=reshape)

    test = DataSet(imgs_test, xq_test, test_labels, dtype=dtype, reshape=reshape)

    return base.Datasets(train=train, validation=validation, test=test)


def load_mdist(data_dir, one_hot):
    return read_data_sets(data_dir, one_hot)
