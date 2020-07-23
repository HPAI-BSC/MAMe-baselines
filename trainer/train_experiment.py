import os
import argparse
import datetime
from PIL import Image
import torch

from consts.paths import Paths
from trainer.utils.consts import Split
from trainer.utils.saver import Saver
from trainer import pipelines as ppl
from trainer.utils.consts import DatasetArgs, PreproArgs, ArchArgs
from trainer.src.training import training

Image.MAX_IMAGE_PIXELS = None

PROJECT_PATH = os.path.abspath(os.path.join(__file__, *(os.path.pardir,)*3))


def main(args):

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = ppl.DATASETS[args.dataset]()
    train_ds = ppl.PREPROCESSES[args.preprocess](Split.TRAIN, *dataset.get_subset(Split.TRAIN))
    val_ds = ppl.PREPROCESSES[args.preprocess](Split.VAL, *dataset.get_subset(Split.VAL))

    input_pipeline = ppl.PIPELINES[args.preprocess](
        datasets_list=[train_ds, val_ds], batch_size=args.batch_size*torch.cuda.device_count(), pin_memory=True if use_cuda else False)

    n_outputs = train_ds.get_n_outputs()
    model = ppl.ARCHITECTURE[args.architecture](num_classes=n_outputs).to(device)

    if torch.cuda.device_count() > 1:
        print("Using {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    loss_function = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

    training_kwargs = {
        "input_pipeline": input_pipeline,
        "model": model,
        "loss_function": loss_function,
        "optimizer": optimizer,
        "device": device,
        "saver": None,
        "retrain": False,
        "max_epochs": args.epochs
    }

    current_date = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
    experiment_name = '{}_{}_{}_{}'.format(args.dataset, args.preprocess, args.architecture, current_date)
    if not args.no_ckpt:
        if args.retrain:
            model_filename = args.retrain
        else:
            model_filename = '{}.ckpt'.format(experiment_name)
        model_path = os.path.join(Paths.models_folder, model_filename)
        training_kwargs['saver'] = Saver(model_path)

    if args.retrain:
        training_kwargs["retrain"] = args.retrain

    if args.preprocess == PreproArgs.LRFS:
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = False

    training(**training_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset.", type=DatasetArgs, choices=list(DatasetArgs))
    parser.add_argument("preprocess", help="Preprocess.", type=PreproArgs, choices=list(PreproArgs))
    parser.add_argument("architecture", help="Architecture.", type=ArchArgs, choices=list(ArchArgs))
    parser.add_argument("batch_size", help="Learning rate.", type=int)
    parser.add_argument("learning_rate", help="Learning rate.", type=float)
    parser.add_argument("epochs", help="Number of epochs to train the model.", type=int)
    parser.add_argument("--retrain", help="Retrain from already existing checkpoint.", type=str, default='')
    parser.add_argument("--no_ckpt", help="Avoid checkpointing.", default=False, action='store_true')
    args = parser.parse_args()

    print(args)
    main(args)
