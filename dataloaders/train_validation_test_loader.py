import os
import numpy as np

# from skimage import io
# from skimage import transform
# from skimage import util

import imageio
from skimage import img_as_float, transform
from skimage.morphology import dilation, disk
import scipy

import torch
from torch.utils import data

from dataloaders.utils import create_or_load_mean, create_distrib_multi_images, normalize_images


class TrainValTestLoader(data.Dataset):

    def __init__(self, mode, dataset, dataset_input_path, num_classes, output_path,
                 model, reference_crop_size, reference_stride_crop,
                 simulate_images=False, mean=None, std=None):

        # Initializing variables.
        self.mode = mode

        self.dataset = dataset
        self.dataset_input_path = dataset_input_path

        self.num_classes = num_classes
        self.output_path = output_path

        self.crop_size = reference_crop_size
        self.stride_crop = reference_stride_crop

        self.data, self.labels, self.names, self.distrib, \
            self.mean, self.std = self.make_dataset(model, mean, std, simulate_images)

        self.num_channels = self.data.shape[-1]

        if len(self.distrib) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

    def load_images(self, areas, simulate_images=False):
        images = []
        masks = []
        names = []

        if simulate_images is True:
            mask = np.random.rand(580, 256, 256, 3)
            images = np.random.rand(580, 256, 256, 3)
            return np.asarray(images), np.asarray(mask)

        for i, area in enumerate(areas):
            names.append('area' + area)

            # print(os.path.join(self.dataset_input_path, stage, 'images', f))
            image = img_as_float(imageio.imread(os.path.join(self.dataset_input_path, 'area' + area +
                                                             '_landsat8_toa_2013_pansharpen.tif')))
            image[np.where(np.isnan(image))] = 0  # replace nan values with 0's
            images.append(image)

            # print(os.path.join(self.dataset_input_path, stage, 'labels', f))
            mask = (imageio.imread(os.path.join(self.dataset_input_path, 'area' + area + '_mask.png'))).astype(int)
            masks.append(mask)

        return np.asarray(images), np.asarray(masks), np.asarray(names)

    def make_dataset(self, model, mean, std, simulate_images):
        assert self.mode in ['Train', 'Test']

        if self.mode == 'Train':
            data, labels, names = self.load_images(['2', '3'], simulate_images=simulate_images)
        else:
            data, labels, names = self.load_images(['4'], simulate_images=simulate_images)
        print(data.shape, labels.shape, names.shape)

        distrib = create_distrib_multi_images(labels, model, self.reference_crop_size,
                                              self.reference_stride_crop, self.num_classes,
                                              filtering_non_classes=(self.dataset == 'road_detection' or
                                                                     self.dataset == 'river'),
                                              percentage_filter=(0.8 if self.dataset == 'road_detection' else 0.99),
                                              percentage_pos_class=(0.1 if self.dataset == 'road_detection' else 0.5))
        print(len(self.distrib))

        if self.mode == 'Train':
            # calculate mean and std if train
            _mean, _std = create_or_load_mean(data, distrib, self.crop_size, self.stride_crop, self.output_path)
        else:
            # set mean and std if test
            _mean, _std = mean, std
        return data, labels, names, distrib, _mean, _std

    def data_augmentation(self, img, label, mask):
        cur_rot = np.random.randint(0, 360)
        possible_rotation = np.random.randint(0, 2)
        if possible_rotation == 1:
            img = scipy.ndimage.rotate(img, cur_rot, order=0, reshape=False)
            label = scipy.ndimage.rotate(label, cur_rot, order=0, reshape=False)
            mask = scipy.ndimage.rotate(mask, cur_rot, order=0, reshape=False)

        # NORMAL NOISE
        possible_noise = np.random.randint(0, 2)
        if possible_noise == 1:
            img = img + np.random.normal(0, 0.01, img.shape)

        # FLIP AUGMENTATION
        flip_random = np.random.randint(0, 3)
        if flip_random == 1:
            img = np.flipud(img)
            label = np.flipud(label)
            mask = np.flipud(mask)
        elif flip_random == 2:
            img = np.fliplr(img)
            label = np.fliplr(label)
            mask = np.fliplr(mask)

        return img, label, mask

    def __getitem__(self, index):

        # Reading items from list.
        cur_map, cur_x, cur_y = self.distrib[index][0], self.distrib[index][1], self.distrib[index][2]

        img = self.data[cur_map, cur_x:cur_x + self.crop_size, cur_y:cur_y + self.crop_size, :]
        label = self.labels[cur_map, cur_x:cur_x + self.crop_size, cur_y:cur_y + self.crop_size]
        mask = np.ones((self.crop_size, self.crop_size), dtype=np.bool)

        # Normalization.
        normalize_images(data, self.mean, self.std)

        if self.mode == 'Train':
            img, label, mask = self.data_augmentation(img, label, mask)

        print(img.shape)
        img = np.transpose(img, (2, 0, 1))

        # Turning to tensors.
        img = torch.from_numpy(img)
        label = torch.from_numpy(label)
        mask = torch.from_numpy(mask)

        # Returning to iterator.
        return img, label, mask, self.distrib[index]

    def __len__(self):
        return len(self.distrib)
