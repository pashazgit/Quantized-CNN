
import pandas as pd
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET


if __name__ == "__main__":

    imageSets_file = os.path.join("/home/mostafa/Downloads/imagenet_object_localization/"
                                  "ILSVRC/ImageSets/CLS-LOC/val.txt")
    val_data = pd.read_csv(imageSets_file, sep=' ', header=None)
    val_data.drop([1], axis=1, inplace=True)
    val_data.columns = ['img_name']

    val_Annotations_dir = os.path.join("/home/mostafa/Downloads/imagenet_object_localization/"
                                       "ILSVRC/Annotations/CLS-LOC/val")
    img_class = []
    for name in tqdm(val_data.img_name):
        name = name + '.xml'
        tree = ET.parse(os.path.join(val_Annotations_dir, name))
        root = tree.getroot()
        img_class.append(root[5][0].text)
    val_data['img_class'] = img_class

    val_data.to_csv(os.path.join("/home/mostafa/Downloads/imagenet_object_localization/"
                                 "ILSVRC/ImageSets/CLS-LOC/val.csv"))
