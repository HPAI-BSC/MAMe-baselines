# MAMe-baselines
This repository contains the code for replicating baseline experiments from our paper: ["A Closer Look at Art Mediums: The MAMe Image Classification Dataset"](https://arxiv.org/abs/2007.13693).
The paper introduces the MAMe dataset: A dataset containing images of high-resolution and variable shape of artworks from 3 different museums:
 - The Metropolitan Museum of Art of New York
 - The Los Angeles County Museum of Art
 - The Cleveland Museum of Art

The MAMe dataset can be downloaded from its [official website](https://hpai.bsc.es/MAMe-dataset).

This code runs with Python 3.7.5. Check the python dependencies in the `requirements.txt` file.

## How to run the baselines
First of all, we have to download the MAMe dataset. Its metadata and image data can be easily downloaded by executing the corresponding downloader bash scripts:
 - Metadata: [download_data](https://github.com/HPAI-BSC/MAMe-baselines/tree/master/dataset/get_metadata.sh)
 - Image data: [download_data](https://github.com/HPAI-BSC/MAMe-baselines/tree/master/dataset/get_data.sh)

```shell
cd dataset
./get_metadata.sh
./get_data.sh
```

Baselines can be trained using the [trainer](https://github.com/HPAI-BSC/MetH-baselines/tree/master/trainer) module.
For example, training the MAMe dataset on its R65k-FS version (i.e. Resolution of 65k pixels with Fixed Shape) from an ImageNet pretrained Vgg11 architecture would be as following:

```shell
python trainer/train_experiment.py MAMe R65k-FS vgg11 512 0.00001 300 r65kfs_vgg11.ckpt --pretrained vgg11-bbd30ac9.pth
```

Arguments required for the `trainer/train_experiment.py` python code can be checked through `--help`:
```shell
> python trainer/train_experiment.py --help
usage: train_experiment.py [-h] [--no_ckpt] [--pretrained PRETRAINED]
                           {MAMe,toy_mame} {R360k-VS,R360k-FS,R65k-VS,R65k-FS}
                           {resnet18,resnet50,vgg11,vgg16,efficientnetb0,efficientnetb3,efficientnetb7,densenet121}
                           batch_size learning_rate epochs ckpt_name

positional arguments:
  {MAMe,toy_mame}       Dataset used for training.
  {R360k-VS,R360k-FS,R65k-VS,R65k-FS}
                        Image online preprocessing.
  {resnet18,resnet50,vgg11,vgg16,efficientnetb0,efficientnetb3,densenet121}
                        Architecture of the neural network.
  batch_size            Batch size: the batch of images will be divided
                        between available GPUs.
  learning_rate         Learning rate used for training.
  epochs                Number of epochs to train the model.
  ckpt_name             Retrain from already existing checkpoint.

optional arguments:
  -h, --help            show this help message and exit
  --no_ckpt             Avoid generating checkpoints.
  --pretrained PRETRAINED
                        Train from a pretrained model.
```

While training, the code saves checkpoints on every epoch in the `model_checkpoints` folder.
Then, we can use one of the checkpoints generated to test the baseline:

```shell
python trainer/test_baseline_model.py MAMe R65k-FS vgg11 128 r65kfs_vgg11.ckpt
```

Alternatively, you can download the official checkpoints we got training the baselines and use them to test:
```shell
./model_checkpoints/get_checkpoints.sh
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
