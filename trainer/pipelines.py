from torchvision.models import resnet18, vgg11, densenet121, resnet50, vgg16

from trainer.src.datasets import MAMeDataset, ToyMAMeDataset
from trainer.src.preprocesses import R65kVSPreprocess, R360kVSPreprocess, R65kFSPreprocess, R360kFSPreprocess
from trainer.src.architectures import efficientnet_b0, efficientnet_b3, efficientnet_b7
from trainer.src.input import InputPipelinePadded, InputPipeline
from trainer.utils.consts import DatasetArgs, PreproArgs, ArchArgs

DATASETS = {
    DatasetArgs.MAME: MAMeDataset,
    DatasetArgs.TOY_MAME: ToyMAMeDataset
}
ARCHITECTURE = {
    ArchArgs.RESNET18: resnet18,
    ArchArgs.RESNET50: resnet50,
    ArchArgs.VGG11: vgg11,
    ArchArgs.VGG16: vgg16,
    ArchArgs.EFFICIENTNETB0: efficientnet_b0,
    ArchArgs.EFFICIENTNETB3: efficientnet_b3,
    ArchArgs.EFFICIENTNETB7: efficientnet_b7,
    ArchArgs.DENSENET121: densenet121,
}
PREPROCESSES = {
    PreproArgs.R360kVS: R360kVSPreprocess,
    PreproArgs.R360kFS: R360kFSPreprocess,
    PreproArgs.R65kVS: R65kVSPreprocess,
    PreproArgs.R65kFS: R65kFSPreprocess,
}
PIPELINES = {
    PreproArgs.R360kVS: InputPipelinePadded,
    PreproArgs.R360kFS: InputPipeline,
    PreproArgs.R65kVS: InputPipelinePadded,
    PreproArgs.R65kFS: InputPipeline,
}
