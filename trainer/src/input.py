import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from trainer.utils.consts import Split


class InputPipeline(object):

    def __init__(self, datasets_list, batch_size=1, num_workers=None, seed=None, pin_memory=False):
        if num_workers is None:
            try:
                num_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
            except KeyError:
                num_workers = 8

        self.seed = seed
        self.dataloaders = {}
        for ds in datasets_list:
            shuffle = ds.subset == Split.TRAIN
            dl = self.get_dataloader(
                ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
            self.dataloaders[ds.subset] = dl

    def _extra_dataloader_kwargs(self, **kwargs):
        if self.seed:
            kwargs['worker_init_fn'] = lambda: np.random.seed(self.seed)
        return kwargs

    def get_dataloader(self, *args, **kwargs):
        kwargs = self._extra_dataloader_kwargs(**kwargs)
        return DataLoader(*args, **kwargs)

    def __getitem__(self, subset):
        try:
            return self.dataloaders[subset]
        except KeyError:
            return None


class InputPipelinePadded(InputPipeline):

    def __init__(self, *args, **kwargs):
        super(InputPipelinePadded, self).__init__(*args, **kwargs)

    def _extra_dataloader_kwargs(self, **kwargs):
        kwargs['collate_fn'] = self.padding_fn
        kwargs = super(InputPipelinePadded, self)._extra_dataloader_kwargs(**kwargs)
        return kwargs

    @staticmethod
    def padding_fn(batch):
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        filenames = None
        try:
            filenames = [item[2] for item in batch]
        except IndexError:
            pass
        max_height = max(img.shape[1] for img in images)
        max_width = max(img.shape[2] for img in images)
        padded_images = []
        for img in images:
            height, width = img.shape[1:]
            h_pad = max_height - height
            w_pad = max_width - width
            top_pad, bottom_pad = h_pad // 2, h_pad // 2
            left_pad, right_pad = w_pad // 2, w_pad // 2
            if h_pad % 2 != 0:
                bottom_pad = h_pad // 2 + 1
            if w_pad % 2 != 0:
                right_pad = w_pad // 2 + 1
            padding_tuple = (left_pad, right_pad, top_pad, bottom_pad)
            padded_img = nn.ZeroPad2d(padding_tuple)(img)
            padded_images.append(padded_img.expand(1, *padded_img.shape))
        images_tensor = torch.cat(padded_images)
        labels_tensor = torch.cat(labels)
        if filenames:
            return images_tensor, labels_tensor, filenames
        else:
            return images_tensor, labels_tensor
