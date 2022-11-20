""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Train a model (single GPU).
"""
import csv
import os
import platform
import time
from logging import addLevelName

import numpy as np
import tensorflow as tf
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

from spleen_dataset.config import dataset_folder as spleen_dataset_folder
from spleen_dataset.dataloader import (SpleenDataloader, SpleenDataset,
                                       get_training_augmentation as get_spleen_training_augmentation,
                                       get_validation_augmentation as get_spleen_validation_augmentation)
from spleen_dataset.utils import get_list_of_patients as get_list_of_spleen_patients, get_split_deterministic as get_spleen_split_deterministic

from prostate_dataset.config import dataset_folder as prostate_dataset_folder
from prostate_dataset.dataloader import (ProstateDataloader, ProstateDataset,
                                       get_training_augmentation as get_prostate_training_augmentation,
                                       get_validation_augmentation as get_prostate_validation_augmentation)
from prostate_dataset.utils import get_list_of_patients as get_list_of_prostate_patients, get_split_deterministic as get_prostate_split_deterministic

from cnn import model


def fitness_calculation(id_num, train_params, layer_dict, net_list, cell_list=None):
    """Train and evaluate a model using evolved parameters.

    Args:
        id_num: string identifying the generation number and the individual number.
        train_params: dictionary with parameters necessary for training
        layer_dict: dict with definitions of the possible layers (name and parameters).
        net_list: list with names of layers defining the network, in the order they appear.
        cell_list: list of predefined cell types that defined a topology (if provided).

    Returns:
        Mean dice coeficient of the model for the last epochs for <initializations> times <folds>-fold cross validation.
    """

    os.environ["TF_SYNC_ON_FINISH"] = "0"
    os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"
    if train_params["log_level"] == "INFO":
        addLevelName(25, "INFO1")
        tf.compat.v1.logging.set_verbosity(25)
    elif train_params["log_level"] == "DEBUG":
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    model_path = os.path.join(train_params["experiment_path"], id_num)
    maybe_mkdir_p(model_path)

    gpus = tf.config.experimental.list_physical_devices("GPU")

    if(len(gpus)):

        gpu_id = int(id_num.split("_")[-1]) % len(gpus)

        tf.config.experimental.set_visible_devices(gpus[gpu_id], "GPU")

        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[gpu_id],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)],
            )
        except RuntimeError as e:
            print(e)

    dataset = train_params["dataset"]
    data_path = train_params["data_path"]
    num_classes = train_params["num_classes"]
    num_channels = train_params["num_channels"]
    skip_slices = train_params["skip_slices"]
    image_size = train_params["image_size"]
    batch_size = train_params["batch_size"]
    epochs = train_params["epochs"]
    eval_epochs = train_params["eval_epochs"]
    num_folds = train_params["folds"]
    num_initializations = train_params["initializations"]
    stem_filters = train_params["stem_filters"]
    max_depth = train_params["max_depth"]

    if(dataset == 'Spleen'):
        patients = get_list_of_spleen_patients(spleen_dataset_folder)
        train_augmentation = get_spleen_training_augmentation(patch_size)
        val_augmentation = get_spleen_validation_augmentation(patch_size)
    elif(dataset == 'Prostate'):
        patients = get_list_of_prostate_patients(prostate_dataset_folder)
        train_augmentation = get_prostate_training_augmentation(patch_size)
        val_augmentation = get_prostate_validation_augmentation(patch_size)
    else:
        raise Exception

    patch_size = (image_size, image_size)

    # Training time start counting here. It needs to be defined outside model_layer(), to make it
    # valid in the multiple calls to segmentation_model.train(). Otherwise, it would be restarted.
    train_params["t0"] = time.time()

    node = platform.uname()[1]

    tf.compat.v1.logging.log(
        level=tf.compat.v1.logging.get_verbosity(),
        msg=f"I am node {node}! Running fitness calculation of {id_num} with "
        f"structure:\n{net_list}",
    )

    val_gen_dice_coef_list = []

    try:
        for initialization in range(num_initializations):
            for fold in range(num_folds):
                net = model.build_net(
                    input_shape=(image_size, image_size, num_channels),
                    num_classes=num_classes,
                    stem_filters=stem_filters,
                    max_depth=max_depth,
                    layer_dict=layer_dict,
                    net_list=net_list,
                    cell_list=cell_list,
                )

                if(dataset == 'Spleen'):
                    train_patients, val_patients = get_spleen_split_deterministic(
                        patients,
                        fold=fold,
                        num_splits=num_folds,
                        random_state=initialization,
                    )

                    train_dataset = SpleenDataset(
                        train_patients, only_non_empty_slices=True, skip_slices=skip_slices
                    )

                    val_dataset = SpleenDataset(val_patients, only_non_empty_slices=True)
                    train_dataloader = SpleenDataloader(
                        train_dataset, batch_size, train_augmentation
                    )

                    val_dataloader = SpleenDataloader(
                        val_dataset, batch_size, val_augmentation
                    )

                elif(dataset == 'Prostate'):
                    train_patients, val_patients = get_prostate_split_deterministic(
                        patients,
                        fold=fold,
                        num_splits=num_folds,
                        random_state=initialization,
                    )
                    
                    train_dataset = ProstateDataset(
                        train_patients, only_non_empty_slices=True, skip_slices=skip_slices
                    )

                    val_dataset = ProstateDataset(val_patients, only_non_empty_slices=True)
                    train_dataloader = ProstateDataloader(
                        train_dataset, batch_size, train_augmentation
                    )

                    val_dataloader = ProstateDataloader(
                        val_dataset, batch_size, val_augmentation
                    )
                else:
                    raise Exception             

                def learning_rate_fn(epoch):
                    initial_learning_rate = 1e-3
                    end_learning_rate = 1e-4
                    power = 0.9
                    return (
                        (initial_learning_rate - end_learning_rate)
                        * (1 - epoch / float(epochs)) ** (power)
                    ) + end_learning_rate

                lr_callback = tf.keras.callbacks.LearningRateScheduler(
                    learning_rate_fn, verbose=False
                )

                history = net.fit(
                    train_dataloader,
                    validation_data=val_dataloader,
                    epochs=epochs,
                    verbose=0,
                    callbacks=[lr_callback],
                )

                val_gen_dice_coef_list.extend(
                    history.history["val_gen_dice_coef"][-eval_epochs:]
                )

                tf.compat.v1.logging.log(
                    level=tf.compat.v1.logging.get_verbosity(),
                    msg=f"[{id_num}] DSC last {eval_epochs} epochs of for training {fold+num_folds*initialization+1}/{num_folds*num_initializations}: {np.mean(history.history['val_gen_dice_coef'][-eval_epochs:])} +- {np.std(history.history['val_gen_dice_coef'][-eval_epochs:])}",
                )

    except Exception as e:
        tf.compat.v1.logging.log(
            level=tf.compat.v1.logging.get_verbosity(),
            msg=f"Exception: {e}",
        )

        return 0

    mean_val_gen_dice_coef = np.mean(val_gen_dice_coef_list)
    std_val_gen_dice_coef = np.std(val_gen_dice_coef_list)
    tf.compat.v1.logging.log(
        level=tf.compat.v1.logging.get_verbosity(),
        msg=f"[{id_num}] DSC: {mean_val_gen_dice_coef} +- {std_val_gen_dice_coef}",
    )

    # save net list as csv (layers)
    net_list_file_path = os.path.join(model_path, "net_list.csv")

    with open(net_list_file_path, mode="w") as f:
        write = csv.writer(f)
        write.writerow(net_list)

    train_params["net"] = net
    train_params["net_list"] = net_list

    return mean_val_gen_dice_coef
