import enum


class Split(enum.Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class DatasetArgs(enum.Enum):
    MAME = 'MAMe'
    TOY_MAME = 'toy_mame'

    def __str__(self):
        return self.value


class PreproArgs(enum.Enum):
    R360kVS = 'R360k-VS'
    R360kFS = 'R360k-FS'
    R65kVS = 'R65k-VS'
    R65kFS = 'R65k-FS'

    def __str__(self):
        return self.value


class ArchArgs(enum.Enum):
    RESNET18 = 'resnet18'
    RESNET50 = 'resnet50'
    VGG11 = 'vgg11'
    VGG16 = 'vgg16'
    EFFICIENTNETB0 = 'efficientnetb0'
    EFFICIENTNETB3 = 'efficientnetb3'
    DENSENET121 = 'densenet121'

    def __str__(self):
        return self.value
