
# Copyright (C), https://github.com/TropComplique/trained-ternary-quantization

import pandas as pd
import os
import shutil
from tqdm import tqdm


def move_tiny_imagenet_train(data_dir):
    """The purpose of this script is to arrange tiny-imagenet data in the following way:
    data_dir/training/n03444034/image_name10.JPEG
    data_dir/training/n03444034/image_name13.JPEG
    data_dir/training/n03444034/image_name15.JPEG
    ...
    data_dir/training/n04067472/image_name16.JPEG
    data_dir/training/n04067472/image_name123.JPEG
    data_dir/training/n04067472/image_name93.JPEG
    ...
    And in the same way arrange validation data.
    """

    # load validation metadata
    annotations_file = os.path.join(data_dir, 'val', 'val_annotations.txt')
    val_data = pd.read_csv(annotations_file, sep='\t', header=None)
    val_data.drop([2, 3, 4, 5], axis=1, inplace=True)  # drop bounding boxes info
    val_data.columns = ['img_name', 'img_class']
    unique_classes = val_data.img_class.unique()

    print('\nmoving training data')

    # create new folders to move data into
    training_dir = os.path.join(data_dir, 'training')
    if not os.path.exists(training_dir):
        os.mkdir(training_dir)
        for name in unique_classes:
            os.mkdir(os.path.join(training_dir, name))

        # loop over all classes
        for name in tqdm(unique_classes):
            # choose images only from a specific class
            class_images = os.listdir(os.path.join(data_dir, 'train', name, 'images'))
            # copy these images to a new folder
            for img in class_images:
                shutil.copyfile(
                    os.path.join(data_dir, 'train', name, 'images', img),
                    os.path.join(training_dir, name, img)
                )

        print('\ntraining data is in', training_dir)

    return training_dir


def move_tiny_imagenet_val(data_dir):
    """The purpose of this script is to arrange tiny-imagenet data in the following way:
    data_dir/training/n03444034/image_name10.JPEG
    data_dir/training/n03444034/image_name13.JPEG
    data_dir/training/n03444034/image_name15.JPEG
    ...
    data_dir/training/n04067472/image_name16.JPEG
    data_dir/training/n04067472/image_name123.JPEG
    data_dir/training/n04067472/image_name93.JPEG
    ...
    And in the same way arrange validation data.
    """

    # load validation metadata
    annotations_file = os.path.join(data_dir, 'val', 'val_annotations.txt')
    val_data = pd.read_csv(annotations_file, sep='\t', header=None)
    val_data.drop([2, 3, 4, 5], axis=1, inplace=True)  # drop bounding boxes info
    val_data.columns = ['img_name', 'img_class']
    unique_classes = val_data.img_class.unique()

    print('\nmoving validation data')

    # create new folders to move the data into
    validation_dir = os.path.join(data_dir, 'validation')
    if not os.path.exists(validation_dir):
        os.mkdir(validation_dir)
        for name in unique_classes:
            os.mkdir(os.path.join(validation_dir, name))

        # loop over all classes
        for name in tqdm(unique_classes):
            # choose images only from a specific class
            class_images = val_data.loc[val_data.img_class == name, 'img_name'].values
            # copy these images to a new folder
            for img in class_images:
                shutil.copyfile(
                    os.path.join(data_dir, 'val', 'images', img),
                    os.path.join(validation_dir, name, img)
                )

        print('\nvalidation data is in', validation_dir)

    return validation_dir

