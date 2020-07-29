# MAMe-baselines
This repository contains the code for replicating baseline experiments from our paper: ["A Closer Look at Art Mediums: The MAMe Image Classification Dataset"](https://arxiv.org/abs/2007.13693).
The paper introduces the MAMe dataset: A dataset containing images of high-resolution and variable shape of artworks from 3 different museums:
 - The Metropolitan Museum of Art of New York
 - The Los Angeles County Museum of Art
 - The Cleveland Museum of Art

The MAMe dataset can be downloaded from its [official website](https://hpai.bsc.es/MAMe-dataset).

This code runs with Python 3.7.2. Check the python dependencies in the `requirements.txt` file.

## How to run the baselines
First of all, we have to download the MAMe dataset. Its metadata and image data can be easily downloaded by executing the corresponding downloader bash scripts:
 - Metadata: [download_data](https://github.com/HPAI-BSC/MAMe-baselines/tree/master/dataset/get_metadata.sh)
 - Image data: [download_data](https://github.com/HPAI-BSC/MAMe-baselines/tree/master/dataset/get_data.sh)

```shell
cd dataset
./get_metadata.sh
./get_data.sh
```

Baselines can be trained and tested using its corresponding experiment scripts inside the [launchers](https://github.com/HPAI-BSC/MetH-baselines/tree/master/launchers) module.
For example, training the MAMe dataset on its LR&FS version (i.e. Low-Resolution and Fixed Shape) for the Vgg11 architecture would be as following:

```shell
./launchers/training/train_LRFS_vgg11.sh
```
While training, the code saves checkpoints on every epoch in the `model_checkpoints` folder.
Then, we can use one of the checkpoints generated to test the baseline:

```shell
python trainer/test_baseline_model.py mame lrfs vgg11 128 model_checkpoints/mame_lrfs_vgg11/mame_lrfs_vgg11_e27.ckpt
```

Alternatively, you can download the official checkpoints we got training the baselines and use them to test:
```shell
./model_checkpoints/get_checkpoints.sh
./launchers/testing/test_LRFS_vgg11.sh
```

## Cite
Please cite our paper if you use this code in your own work:
```
@article{pares2020mame,
    title={A Closer Look at Art Mediums: The MAMe Image Classification Dataset},
    author={Par{\'e}s, Ferran and Arias-Duart, Anna and Garcia-Gasulla, Dario and Campo-Franc{\'e}s, Gema and Viladrich, Nina and Labarta, Jes{\'u}s and Ayguad{\'e}, Eduard},
    journal={arXiv preprint arXiv:2007.13693},
    year={2020}
}
```

## About
This work is currently under review process at the International Journal of Computer Vision.

If you have any questions about this code, please contact: Ferran Parés <ferran.pares@bsc.es>
