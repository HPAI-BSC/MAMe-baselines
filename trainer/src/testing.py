from typing import List

import torch
import numpy as np

from trainer.utils.consts import Split
from trainer.utils.pytorch_utils import squeeze_generic


def testing(input_pipeline, model, device):
    model.eval()

    # Test epoch
    pred_labels = []
    true_labels = []
    image_filenames = []
    with torch.no_grad():
        for batch_images, batch_labels, batch_filenames in input_pipeline[Split.TEST]:
            # Loading tensors in the used device
            step_images = batch_images.to(device)

            step_output = model(step_images)
            pred_labels += squeeze_generic(step_output, axes_to_keep=[0]).tolist()
            true_labels += squeeze_generic(batch_labels, axes_to_keep=[0]).tolist()
            image_filenames += list(batch_filenames)

        pred_labels = np.argmax(pred_labels, axis=1).tolist()
    return pred_labels, true_labels, image_filenames


def accuracy(pred_values: List[int], true_values: List[int]):
    correct = np.sum(np.array(pred_values) == np.array(true_values))
    return correct/len(pred_values)
