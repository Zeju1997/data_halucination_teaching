import skimage.io as io
import matplotlib.pylab as plt
import numpy as np
from utils import mhd

from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from medpy.io import load
import random

import torch
import os
from PIL import Image  # using pillow-simd for increased speed

import cv2
from torchvision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert("L")

class Retouch_dataset(Dataset):
    def __init__(self,
                 base_dir,
                 list_dir,
                 size=(512, 512),
                 split='train',
                 is_train=False,
                 transform=None,
                 ext='.png'):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir,
                                             self.split + '.txt')).readlines()
        self.data_dir = base_dir
        self.loader = pil_loader
        self.to_tensor = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                # transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ])

        self.is_train = is_train
        self.transform = transform
        self.ext = ext

    def augment(self, data, label):
        data_label = torch.cat((data, label), dim=0)
        data_label_aug = self.transform(data_label)
        data_aug = data_label_aug[0, :, :].unsqueeze(0)
        label_aug = data_label_aug[1, :, :].unsqueeze(0)
        return data_aug, label_aug

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_name = self.sample_list[idx].strip('\n')

        vendor = sample_name.split(' ')[0]
        slice_name = sample_name.split(' ')[1]
        slice_idx = sample_name.split(' ')[2].zfill(3)

        data_path = os.path.join(self.data_dir,
                                 vendor,
                                 slice_name,
                                 'images',
                                 slice_idx + self.ext)

        label_path = os.path.join(self.data_dir,
                                  vendor,
                                  slice_name,
                                  'labels',
                                  slice_idx + '.npy')

        data = self.to_tensor(self.loader(data_path))
        label = torch.from_numpy(np.load(label_path))
        label_idx = torch.argmax(label, dim=0, keepdim=True)

        transform_avaliable = self.transform is not None and self.is_train
        do_aug = transform_avaliable and random.random() > 0.5

        if do_aug:
            data, label_idx = self.augment(data, label_idx)

        label_idx = label_idx.squeeze(0).long()

        sample = {'image': data,
                  'label': label_idx,
                  'case_name': sample_name}
        # print((label_idx==0).sum()/512**2)
        return sample

# Test Unit
# flip = transforms.RandomHorizontalFlip(p=0.5)
# base_dir = 'Retouch-dataset_test/pre_processed/'
# list_dir = ''
# dataset = Retouch_dataset(base_dir, list_dir, transform=flip)
# l = dataset[3]['label']
# d = dataset[3]['image']
#
# print(l.shape, d.shape)
#
# img = d.permute(1, 2, 0).numpy()
# print((img[:, :, 0] == img[:, :, 2]).all())
# print(dataset[3]['case_name'])
# plt.figure()
# plt.imshow(img)
