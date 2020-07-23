import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from trainer.utils.consts import Split


class OriginalDataset(Dataset):

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


class DownsampledDataset(OriginalDataset):

    def __init__(self, *args, **kwargs):
        try:
            size = kwargs['size']
            del kwargs['size']
        except KeyError:
            size = (224, 224)
        assert size[0] < 256 and size[1] < 256
        self.size = size
        super(DownsampledDataset, self).__init__(*args, **kwargs)

    def _get_transforms_list(self):
        if self.subset == Split.TRAIN:
            return_transform = [
                transforms.Resize((256, 256)),
                transforms.RandomRotation(degrees=30),
                transforms.RandomCrop(self.size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        else:
            return_transform = [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        return return_transform


class DownRatioDataset(OriginalDataset):

    def __init__(self, *args, down_ratio=0.2):
        assert 0.0 < down_ratio < 1.0
        self.down_ratio = down_ratio
        super().__init__(*args)

    def _custom_img_transformation(self, image):
        if self.down_ratio is not None:
            width, height = image.size
            down_w, down_h = round(width*self.down_ratio), round(height*self.down_ratio)
            image = image.resize((down_w, down_h), Image.ANTIALIAS)
        return image


class DownAxisDataset(OriginalDataset):
    def __init__(self, *args, **kwargs):
        try:
            min_axis_size = kwargs['min_axis_size']
            del kwargs['min_axis_size']
        except KeyError:
            min_axis_size = 500
        self.min_axis_size = min_axis_size
        super().__init__(*args, **kwargs)

    @staticmethod
    def _random_crop_fraction(img):
        width, height = img.size
        target_width, target_height = int(width*0.875), int(height*0.875)
        left = random.randint(0, width-target_width)
        top = random.randint(0, height-target_height)
        return TF.crop(img, top, left, target_height, target_width)

    @staticmethod
    def _center_crop_fraction(img):
        width, height = img.size
        target_width, target_height = width*0.875, height*0.875
        left = (width-target_width)//2
        top = (height-target_height)//2
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
        if orig_height < orig_width:  # Landscape
            target_height = self.min_axis_size
            down_ratio = target_height/orig_height
            target_width = round(orig_width * down_ratio)
        else:  # Portrait (or squared)
            target_width = self.min_axis_size
            down_ratio = target_width/orig_width
            target_height = round(orig_height * down_ratio)

        image = image.resize((target_width, target_height), Image.ANTIALIAS)
        return image
