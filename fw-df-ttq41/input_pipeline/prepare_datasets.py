
import sys
sys.path.insert(0, "./inq-ttq-offline/input_pipeline")
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from input_pipeline.prepare_folders import move_tiny_imagenet_train, move_tiny_imagenet_val


def create_train_dataset(config):
    """
    Build an input pipeline for training and evaluation.
    For training data it does data augmentation.
    """

    TRAIN_DIR = move_tiny_imagenet_train(config.data_dir)
    """It assumes that training image data is in the following form:
    TRAIN_DIR/class4/image44.jpg
    TRAIN_DIR/class4/image12.jpg
    ...
    TRAIN_DIR/class55/image33.jpg
    TRAIN_DIR/class55/image543.jpg
    ...
    TRAIN_DIR/class1/image6.jpg
    TRAIN_DIR/class1/image99.jpg
    ...
    And the same for validation data.
    """

    # training data augmentation on the fly
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # mean and std are taken from here:
    # http://pytorch.org/docs/master/torchvision/models.html

    train_dataset = ImageFolder(TRAIN_DIR, train_transform)
    return train_dataset


def create_val_dataset(config):
    """
    Build an input pipeline for training and evaluation.
    For training data it does data augmentation.
    """

    VAL_DIR = move_tiny_imagenet_val(config.data_dir)
    """It assumes that training image data is in the following form:
    TRAIN_DIR/class4/image44.jpg
    TRAIN_DIR/class4/image12.jpg
    ...
    TRAIN_DIR/class55/image33.jpg
    TRAIN_DIR/class55/image543.jpg
    ...
    TRAIN_DIR/class1/image6.jpg
    TRAIN_DIR/class1/image99.jpg
    ...
    And the same for validation data.
    """

    # for validation data
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # mean and std are taken from here:
    # http://pytorch.org/docs/master/torchvision/models.html

    val_dataset = ImageFolder(VAL_DIR, val_transform)
    return val_dataset
