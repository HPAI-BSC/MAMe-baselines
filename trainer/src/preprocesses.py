import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from trainer.utils.consts import Split


class OriginalPreprocess(Dataset):

    def __init__(self, subset, image_paths, image_labels, include_filename=False):
        assert subset not in [attr for attr in dir(Split) if not attr.startswith('__')]
        self.subset = subset
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.include_filename = include_filename
        self.transform = transforms.Compose(self._get_transforms_list())
        self.label_transformation = None

        self.n_outputs = len(set(image_labels))
        self.set_up_label_transformation_for_classification()

    def set_up_label_transformation_for_classification(self):
        sorted_labels = sorted(set(self.image_labels))
        label2idx = {raw_label: idx for idx, raw_label in enumerate(sorted_labels)}
        self.label_transformation = lambda x: torch.tensor(label2idx[x], dtype=torch.long).view(-1, 1)

    def _get_transforms_list(self):
        return [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]

    def get_n_outputs(self):
        return self.n_outputs

    def __len__(self):
        return len(self.image_paths)

    def _custom_img_transformation(self, image):
        return image

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self._custom_img_transformation(image)

        label = self.image_labels[idx]

        if self.label_transformation:
            label = self.label_transformation(label)
        image = self.transform(image)

        if self.include_filename:
            return image, label, os.path.basename(img_path)
        else:
            return image, label


class DownsampledPreprocess(OriginalPreprocess):

    def __init__(self, *args, **kwargs):
        try:
            self.image_size = kwargs['size']
            del kwargs['size']
        except KeyError:
            self.image_size = (256, 256)
        self.crop_size = (int(self.image_size[0] * 0.875), int(self.image_size[1] * 0.875))
        super().__init__(*args, **kwargs)

    def _get_transforms_list(self):
        if self.subset == Split.TRAIN:
            return_transform = [
                transforms.Resize(self.image_size),
                transforms.RandomRotation(degrees=30),
                transforms.RandomCrop(self.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        else:
            return_transform = [
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        return return_transform


class ReduceKeepARPreprocess(OriginalPreprocess):
    def __init__(self, *args, **kwargs):
        try:
            target_pixels = kwargs['target_pixels']
            del kwargs['target_pixels']
        except KeyError:
            target_pixels = 256 * 256  # 65_536
        self.target_pixels = target_pixels
        super().__init__(*args, **kwargs)

    @staticmethod
    def _random_crop_fraction(img):
        width, height = img.size
        target_width, target_height = int(width * 0.875), int(height * 0.875)
        left = random.randint(0, width - target_width)
        top = random.randint(0, height - target_height)
        return TF.crop(img, top, left, target_height, target_width)

    @staticmethod
    def _center_crop_fraction(img):
        width, height = img.size
        target_width, target_height = width * 0.875, height * 0.875
        left = (width - target_width) // 2
        top = (height - target_height) // 2
        return TF.crop(img, top, left, target_height, target_width)

    def _get_transforms_list(self):
        if self.subset == Split.TRAIN:
            return_transform = [
                transforms.RandomRotation(degrees=30),
                transforms.Lambda(self._random_crop_fraction),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        else:
            return_transform = [
                transforms.Lambda(self._center_crop_fraction),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        return return_transform

    def _custom_img_transformation(self, image):
        orig_width, orig_height = image.size
        ar = orig_width / orig_height
        target_width = int(np.sqrt(self.target_pixels * ar))
        target_height = int(np.sqrt(self.target_pixels / ar))

        image = image.resize((target_width, target_height), Image.ANTIALIAS)
        return image


class R65kVSPreprocess(ReduceKeepARPreprocess):
    def __init__(self, *args, **kwargs):
        self.target_pixels = 256 * 256
        super().__init__(*args, **kwargs)


class R360kVSPreprocess(ReduceKeepARPreprocess):
    def __init__(self, *args, **kwargs):
        self.target_pixels = 600 * 600
        super().__init__(*args, **kwargs)


class R65kFSPreprocess(DownsampledPreprocess):
    def __init__(self, *args, **kwargs):
        kwargs['size'] = (256, 256)
        super().__init__(*args, **kwargs)


class R360kFSPreprocess(DownsampledPreprocess):
    def __init__(self, *args, **kwargs):
        kwargs['size'] = (600, 600)
        super().__init__(*args, **kwargs)
