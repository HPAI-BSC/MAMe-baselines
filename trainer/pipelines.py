from torchvision.models import vgg11

from trainer.src.datasets import MAMeDataset, ToyMAMeDataset
from trainer.src.preprocessing import DownAxisDataset, DownsampledDataset
from trainer.src.architectures import resnet18_nobn
from trainer.src.input import InputPipelinePadded, InputPipeline
from trainer.utils.consts import DatasetArgs, PreproArgs, ArchArgs

DATASETS = {
    DatasetArgs.MAME: MAMeDataset,
    DatasetArgs.TOY_MAME: ToyMAMeDataset
}
ARCHITECTURE = {
    ArchArgs.RESNET18: resnet18_nobn,
    ArchArgs.VGG11: vgg11
}
PREPROCESSES = {
    PreproArgs.HRVS: DownAxisDataset,
    PreproArgs.LRFS: DownsampledDataset
}
PIPELINES = {
    PreproArgs.HRVS: InputPipelinePadded,
    PreproArgs.LRFS: InputPipeline
}
