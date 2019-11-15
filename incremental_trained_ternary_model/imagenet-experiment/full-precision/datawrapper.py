
from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import h5py


class train_dataset(Dataset):

    def __init__(self, hdf5_file_path):
        self.hdf5_file = h5py.File(hdf5_file_path, mode='r')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             normalize])

    def __len__(self):
        return len(self.hdf5_file["train_images"])

    def __getitem__(self, idx):
        image = self.hdf5_file["train_images"][idx]
        image = self.transform(image)
        label = self.hdf5_file["train_labels"][idx]
        return image, label


class val_dataset(Dataset):

    def __init__(self, hdf5_file_path):
        self.hdf5_file = h5py.File(hdf5_file_path, mode='r')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             normalize])

    def __len__(self):
        return len(self.hdf5_file["val_images"])

    def __getitem__(self, idx):
        image = self.hdf5_file["val_images"][idx]
        image = self.transform(image)
        label = self.hdf5_file["val_labels"][idx]
        return image, label
