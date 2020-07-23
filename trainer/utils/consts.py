import enum


class Split(enum.Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class DatasetArgs(enum.Enum):
    MAME = 'mame'
    TOY_MAME = 'toy_mame'

    def __str__(self):
        return self.value


class PreproArgs(enum.Enum):
    HRVS = 'hrvs'
    LRFS = 'lrfs'

    def __str__(self):
        return self.value


class ArchArgs(enum.Enum):
    RESNET18 = 'resnet18'
    VGG11 = 'vgg11'

    def __str__(self):
        return self.value
