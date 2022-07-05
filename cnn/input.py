""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Input function and dataset info classes.

    References:
    https://github.com/tensorflow/models/blob/r1.10.0/tutorials/image/cifar10_estimator/cifar10.py

"""

import os
import platform
from turtle import width

import numpy as np
import psutil
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()


class DataSet(object):
    def __init__(self, data_info, data_aug, subtract_mean, process_for_training):
        """Initialize DataSet base class.

        Args:
            data_info: only PascalVOCInfo object supported so far.
            data_aug: (bool) True if user wants to train using data augmentation.
            subtract_mean: (bool) True if calculated mean on the training set should be
                subtracted.
            process_for_training: (bool) if True, the dataset is processed for training (include
                shuffle and dataset augmentation); for validation and test, set it to False.
        """

        self.data_augmentation = data_aug
        self.info = data_info
        self.process_for_training = process_for_training
        self.subtract_mean = subtract_mean

    def get_file_list(self, dataset_type):
        """Get the file list corresponding to the *mode*.

        Args:
            dataset_type: (str) one of 'train', 'valid' or 'test'.

        Returns:
            list of files with examples.
        """

        if dataset_type == "train":
            return self.info.train_files
        elif dataset_type == "valid":
            return self.info.valid_files
        elif dataset_type == "test":
            return self.info.test_files

    def _one_hot_encode_mask(self, mask):
        """One hot encode the mask tensor.

        Args:
            mask: tensor tf.uint8 with shape = [height, width, 1] and pallete encoding of classes.

        Returns:
            mask  (tf.float32 [0, 1] and shape = [height, width, classes]).
        """
        # create a binary mask for each channel (class)
        one_hot_mask = []
        for _class in range(self.info.num_classes):
            class_mask = tf.reduce_all(tf.equal(mask, _class), axis=-1)
            one_hot_mask.append(class_mask)
        one_hot_mask = tf.stack(one_hot_mask, axis=-1)
        one_hot_mask = tf.cast(one_hot_mask, tf.float32)

        # mask (height, width, num_classes)
        return one_hot_mask

    def rec_parser(self, serialized_example):
        """Parse a single tf.Example into image and mask tensors.

        Args:
            serialized_example: tfrecords example.

        Returns:
            image (tf.float32 [0, 1] and shape = [height, width, channels]).
            mask  (tf.float32 [0, 1] and shape = [height, width, classes]).
        """

        features = tf.compat.v1.parse_single_example(
            serialized_example,
            features={
                "height": tf.compat.v1.FixedLenFeature([], tf.int64),
                "width": tf.compat.v1.FixedLenFeature([], tf.int64),
                "channels": tf.compat.v1.FixedLenFeature([], tf.int64),
                "mask_raw": tf.compat.v1.FixedLenFeature([], tf.string),
                "image_raw": tf.compat.v1.FixedLenFeature([], tf.string),
            },
        )

        image = tf.compat.v1.decode_raw(features["image_raw"], tf.uint8)
        image = tf.reshape(
            image, (features["height"], features["width"], features["channels"])
        )

        mask = tf.compat.v1.decode_raw(features["mask_raw"], tf.uint8)
        mask = tf.reshape(mask, (features["height"], features["width"], 1))

        # Rescale the values of the image and the mask from the range [0, 255] to [0, 1.0]
        image = tf.divide(tf.cast(image, tf.float32), 255.0)

        # Make one hot encoding of the mask (height, width) -> (height, width, num_classes)
        mask = self._one_hot_encode_mask(mask)

        # Subtract mean_img from image
        if self.subtract_mean:
            image = tf.subtract(image, self.info.mean_image, name="mean_subtraction")

        image, mask = self.preprocess(image, mask)

        return image, mask

    def make_batch(self, batch_size, dataset_type, threads=0):
        """Read the images and masks from files corresponding to *dataset_type* and prepare a
            batch of them.

        Args:
            batch_size: (int) size of the batch.
            dataset_type: (str) one of 'train', 'valid' or 'test'.
            threads: (int) number of threads to parse dataset examples.

        Returns:
            images (shape = (batch_size, height, width, channels)).
            masks (shape = (batch_size, height, width, classes)).
        """

        if not threads:
            if os.uname().sysname == "Linux":
                threads = len(os.sched_getaffinity(0))
            elif platform.system() == "Windows":
                threads = len(psutil.Process().cpu_affinity())
            else:
                threads = os.cpu_count()

        dataset = tf.data.TFRecordDataset(self.get_file_list(dataset_type))
        dataset = dataset.map(self.rec_parser, num_parallel_calls=threads)
        dataset = dataset.prefetch(8 * batch_size)

        # For training, shuffle dataset and repeat forever.
        if self.process_for_training:
            # Ensure that the capacity is sufficiently large to provide good random shuffling.
            buffer_size = int(0.4 * self.info.num_train_ex)
            # Shuffle dataset every new iteration (epoch)
            dataset = dataset.shuffle(
                buffer_size=buffer_size, reshuffle_each_iteration=True
            )
            dataset = dataset.repeat()

        # Batch results by up to batch_size, and then fetch the tuple from the iterator.
        batched_dataset = dataset.batch(batch_size)
        iterator = tf.compat.v1.data.make_one_shot_iterator(batched_dataset)

        images, masks = iterator.get_next()

        return images, masks

    def preprocess(self, image, mask):
        """Resize an image with shape = [height, width, channels] and a mask with shape = [height, width, classes].
        Args:
            image: raw image (tf.float32 [0, 1] and shape = [height, width, channels]).
            mask: raw mask (tf.float32 [0, 1] and shape = [height, width, num_classes]).

        Returns:
            preprocessed image, with same shape.
        """
        tf.compat.v1.logging.log(
            level=tf.compat.v1.logging.get_verbosity(),
            msg=f"image/mask size: {self.info.height}, {self.info.width}",
        )

        image = tf.compat.v1.image.resize(image, (self.info.height, self.info.width))
        mask = tf.compat.v1.image.resize(mask, (self.info.height, self.info.width))

        # TODO augmentation for image and masks
        # image = tf.image.random_flip_left_right(image)
        if self.data_augmentation:
            pass

        return image, mask


class PascalVOC12Info(object):
    def __init__(self, data_path, validation=True):
        """Pascal VOC 2012 dataset information.

            Info in http://host.robots.ox.ac.uk/pascal/VOC/voc2012/.

        Args:
            data_path: (str) path to the folder containing the tfrecords files.
            validation: (bool) whether to use the validation dataset for validation.
        """

        self.data_path = data_path
        #self.height = 224  # after preprocessing
        #self.width = 224  # after preprocessing
        self.num_channels = 3
        self.mean_image = np.load(
            os.path.join(self.data_path, "pascalvoc12_train_mean.npz")
        )["train_img_mean"]
        self.std_image = np.load(
            os.path.join(self.data_path, "pascalvoc12_train_std.npz")
        )["train_img_std"]
        self.num_classes = 21
        # self.pad = 4

        self.train_files = [
            os.path.join(self.data_path, f)
            for f in os.listdir(self.data_path)
            if f.startswith("train")
        ]
        self.valid_files = [
            os.path.join(self.data_path, f)
            for f in os.listdir(self.data_path)
            if f.startswith("valid")
        ]
        self.test_files = [
            os.path.join(self.data_path, f)
            for f in os.listdir(self.data_path)
            if f.startswith("test")
        ]

        # if user wants to train using all images
        if not validation:
            self.train_files = self.train_files + self.valid_files
            self.valid_files = []

        self.num_train_ex = count_records(self.train_files)
        self.num_valid_ex = count_records(self.valid_files)
        self.num_test_ex = count_records(self.test_files)


def count_records(file_list):
    """Count total number of records in a file list.

    Args:
        file_list: list of tfrecords files.

    Returns:
        total number of records in the file list.
    """

    c = 0
    for file_path in file_list:
        for _ in tf.compat.v1.python_io.tf_record_iterator(file_path):
            c += 1
    return c


def input_fn(
    data_info,
    dataset_type,
    batch_size,
    data_aug,
    subtract_mean,
    process_for_training,
    threads=0,
):
    """Create input function for model.

    Args:
        data_info: PascalVOC12 supported.
        dataset_type: (str) one of 'train', 'valid' or 'test'.
        batch_size: (int) number of examples in a batch (can be different for train or evaluate)
        data_aug: (bool) True if user wants to train using data augmentation.
        subtract_mean: (bool) True if calculated mean on the training set should be
            subtracted.
        process_for_training: (bool) if True, the dataset is processed for training (shuffle and
            repeat, for example); for validation and test, set it to False.
        threads: (int) number of threads to read and batch dataset; if 0, it set to the number
            of logical cores.

    Returns:
        batch of images (shape = (batch_size, height, width, channels)).
        batch of masks (shape = (batch_size, height, width, classes)).
    """

    with tf.device("/cpu:0"):
        dataset = DataSet(data_info, data_aug, subtract_mean, process_for_training)
        image_batch, mask_batch = dataset.make_batch(batch_size, dataset_type, threads)

        return image_batch, mask_batch
