import os
import argparse
import pandas as pd
from PIL import Image
import torch

from consts.paths import Paths
from trainer.utils.consts import Split, ArchArgs
from trainer.utils.saver import load_checkpoint
from trainer.utils.utils import accuracy
from trainer import pipelines as ppl
from trainer.utils.consts import DatasetArgs, PreproArgs
from trainer.src.testing import testing

Image.MAX_IMAGE_PIXELS = None

PROJECT_PATH = os.path.abspath(os.path.join(__file__, *(os.path.pardir,) * 2))


def main(args):
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = ppl.DATASETS[args.dataset]()
    test_ds = ppl.PREPROCESSES[args.preprocess](Split.TEST, *dataset.get_subset(Split.TEST), include_filename=True)

    input_pipeline = ppl.PIPELINES[args.preprocess](
        datasets_list=[test_ds], batch_size=args.batch_size, pin_memory=True if use_cuda else False)

    n_outputs = test_ds.get_n_outputs()
    model = ppl.ARCHITECTURE[args.architecture](num_classes=n_outputs).to(device)

    if torch.cuda.device_count() > 1:
        print("Using {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    model, _, epoch = load_checkpoint(args.model_ckpt, model)

    if args.preprocess in [PreproArgs.R65kFS, PreproArgs.R360kFS]:
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = False

    pred_labels, true_labels, image_filenames = testing(input_pipeline, model, device)

    name_model = os.path.splitext(os.path.basename(args.model_ckpt))[0]
    output_path = os.path.join(Paths.inference_results_folder, "{}_e{}.csv".format(name_model, epoch))
    out_df = pd.DataFrame({"True labels": true_labels, "Predicted labels": pred_labels, "Filenames": image_filenames})
    out_df.to_csv(output_path, header=True, index=False)

    print(accuracy(pred_labels, true_labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset.", type=DatasetArgs, choices=list(DatasetArgs))
    parser.add_argument("preprocess", help="Preprocess.", type=PreproArgs, choices=list(PreproArgs))
    parser.add_argument("architecture", help="Architecture.", type=ArchArgs, choices=list(ArchArgs))
    parser.add_argument("batch_size", help="Learning rate.", type=int)
    parser.add_argument("model_ckpt", help="Path to checkpoint.", type=str, default='')
    args = parser.parse_args()

    assert args.dataset in ppl.DATASETS
    assert args.preprocess in ppl.PREPROCESSES

    main(args)
