import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image, ImageFile
# from dataset.autoaugment import ImageNetPolicy
import imageio
ImageFile.LOAD_TRUNCATED_IMAGES = True


class SynthiaDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(256, 256), mean=(128, 128, 128), ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        '''
        self.id_to_trainid = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
                              15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12,
                              8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18}
        '''
        self.id_to_trainid = {3: 1, 4: 2, 2: 3, 21: 4, 5: 5, 7: 6,
                              15: 7, 9: 8, 6: 9, 16: 10, 1: 11, 10: 12, 17: 13,
                              8: 14, 18: 15, 19: 16, 20: 17, 12: 18, 11: 19}
        self.ignore_label = 0

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "RGB/%s" % name)
            label_file = osp.join(self.root, "GT/LABELS/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        name = self.img_ids[index]

        image = Image.open(osp.join(self.root, "images/%s" % name)).convert('RGB')

        # original data labels
        # label = np.asarray(imageio.imread(osp.join(self.root, "labels/%s" % name), format='PNG-FI'))[:,:,0]  # uint16
        # label = Image.fromarray(label)

        label = Image.open(osp.join(self.root, "labels/%s" % name))
        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.int8)

        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
        # for k, v in self.id2label.items():
            label_copy[label == k] = v
        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        sample = {'image': image.copy(),
                  'label': label_copy.copy()}

        return sample
