import os
import argparse
from PIL import Image
import torch

from consts.paths import Paths
from trainer.utils.consts import Split
from trainer.utils.saver import Saver
from trainer import pipelines as ppl
from trainer.utils.consts import DatasetArgs, PreproArgs, ArchArgs
from trainer.src.training import training
from trainer.utils.saver import load_checkpoint_pretrained

Image.MAX_IMAGE_PIXELS = None

PROJECT_PATH = os.path.abspath(os.path.join(__file__, *(os.path.pardir,)*3))


def main(args):

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = ppl.DATASETS[args.dataset]()
    train_ds = ppl.PREPROCESSES[args.preprocess](Split.TRAIN, *dataset.get_subset(Split.TRAIN), **extra_kwargs)
    val_ds = ppl.PREPROCESSES[args.preprocess](Split.VAL, *dataset.get_subset(Split.VAL), **extra_kwargs)

    input_pipeline = ppl.PIPELINES[args.preprocess](
        datasets_list=[train_ds, val_ds], batch_size=args.batch_size, pin_memory=True if use_cuda else False)

    n_outputs = train_ds.get_n_outputs()
    model = ppl.ARCHITECTURE[args.architecture](num_classes=n_outputs)

    if args.pretrained:
        model_filename = args.pretrained
        model_path = os.path.join(Paths.models_folder, model_filename)
        model = load_checkpoint_pretrained(model_path, model)

    model = model.to(device)

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

    model_path = os.path.join(Paths.models_folder, args.ckpt_name)
    if not args.no_ckpt:
        training_kwargs['saver'] = Saver(model_path)

    if os.path.exists(model_path):
        training_kwargs["retrain"] = args.ckpt_name

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
    parser.add_argument("ckpt_name", help="Retrain from already existing checkpoint.", type=str)
    parser.add_argument("--no_ckpt", help="Avoid checkpointing.", default=False, action='store_true')
    parser.add_argument("--pretrained", help="Train from pretrained model.", type=str, default=False)
    args = parser.parse_args()

    assert args.dataset in ppl.DATASETS
    assert args.preprocess in ppl.PREPROCESSES

    print(args)
    main(args)
