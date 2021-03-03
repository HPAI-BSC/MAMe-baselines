import os
import pandas as pd

from trainer.utils.consts import Split
from trainer.utils.paths import Paths


class MAMeDataset:
    csv_path = Paths.MAMe.csv_path
    data_folder = Paths.MAMe.images_path

    def __init__(self):
        self.df = pd.read_csv(self.csv_path)

    def get_subset(self, subset):
        assert subset in Split
        df_subset = self.df[self.df['Subset'] == subset.value]
        image_filenames = df_subset['Image file'].tolist()
        image_labels = df_subset['Medium'].tolist()
        image_filepaths = [os.path.join(self.data_folder, img_filename) for img_filename in image_filenames]
        return image_filepaths, image_labels


class ToyMAMeDataset(MAMeDataset):
    csv_path = Paths.MAMe.toy_csv_path
