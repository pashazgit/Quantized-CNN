
import pandas as pd
import os
import shutil
from tqdm import tqdm
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import h5py


if __name__ == "__main__":

    hdf5_file = h5py.File("/home/mostafa/Downloads/imagenet_trva_2012.h5", mode='w')

    imageSets_file = os.path.join("/home/mostafa/Downloads/imagenet_object_localization/"
                                  "ILSVRC/ImageSets/CLS-LOC/val.csv")
    val_data = pd.read_csv(imageSets_file)
    unique_classes = val_data.img_class.unique()

    # ---------------------------------------------------------------------------------
    # Create validation dataset
    val_cmposed = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224)])
    # len_val = len(val_data.img_name)
    len_val = 0
    for img in tqdm(val_data.img_name):
        img = img + ".JPEG"
        img_cont = np.array(Image.open(os.path.join("/home/mostafa/Downloads/imagenet_object_localization/"
                                                    "ILSVRC/Data/CLS-LOC/val", img)))
        if len(img_cont.shape) == 3 and img_cont.shape[2] == 3:
            len_val += 1

    hdf5_file.create_dataset("val_images", (len_val, 224, 224, 3), np.uint8)
    # hdf5_file.create_dataset("val_labels", (len_val,), np.uint16)

    i = 0
    class_label = 0
    labels = []
    for class_name in tqdm(unique_classes):
        class_images = val_data.loc[val_data.img_class == class_name, 'img_name'].values

        for img in class_images:
            img = img + ".JPEG"
            img_cont = Image.open(os.path.join("/home/mostafa/Downloads/imagenet_object_localization/"
                                               "ILSVRC/Data/CLS-LOC/val", img))
            if len(np.array(img_cont).shape) == 3 and np.array(img_cont).shape[2] == 3:
                hdf5_file["val_images"][i, ...] = val_cmposed(img_cont)
                labels.append(class_label)
                i += 1

        class_label += 1

    hdf5_file.create_dataset("val_labels", data=np.array(labels))

    # --------------------------------------------------------------------------------
    # Create train dataset
    tr_cmposed = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip()])
    len_tr = 0
    for class_name in tqdm(unique_classes):
        class_images = os.listdir(os.path.join("/home/mostafa/Downloads/imagenet_object_localization/"
                                               "ILSVRC/Data/CLS-LOC/train", class_name))
        for img in class_images:
            img_cont = np.array(Image.open(os.path.join("/home/mostafa/Downloads/imagenet_object_localization/"
                                               "ILSVRC/Data/CLS-LOC/train", class_name, img)))
            if len(img_cont.shape) == 3 and img_cont.shape[2] == 3:
                len_tr += 1

    hdf5_file.create_dataset("train_images", (len_tr, 224, 224, 3), np.uint8)
    # hdf5_file.create_dataset("train_labels", (len_tr,), np.uint16)

    i = 0
    class_label = 0
    labels = []
    for class_name in tqdm(unique_classes):
        class_images = os.listdir(os.path.join("/home/mostafa/Downloads/imagenet_object_localization/"
                                               "ILSVRC/Data/CLS-LOC/train", class_name))
        for img in class_images:
            img_cont = Image.open(os.path.join("/home/mostafa/Downloads/imagenet_object_localization/"
                                                        "ILSVRC/Data/CLS-LOC/train", class_name, img))
            if len(np.array(img_cont).shape) == 3 and np.array(img_cont).shape[2] == 3:
                hdf5_file["train_images"][i, ...] = tr_cmposed(img_cont)
                labels.append(class_label)
                i += 1

        class_label += 1

    hdf5_file.create_dataset("train_labels", data=np.array(labels))

    # -------------------------------------------------------------------------------
    hdf5_file.close()


