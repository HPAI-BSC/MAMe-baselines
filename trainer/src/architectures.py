import torch
from torch import nn
from torchvision.models.resnet import ResNet, BasicBlock, load_state_dict_from_url, model_urls


class ResNet_noBN(ResNet):
    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            self.deactivate_batchnorm(m)

    @staticmethod
    def deactivate_batchnorm(m):
        if isinstance(m, nn.BatchNorm2d):
            m.reset_parameters()
            m.eval()
            with torch.no_grad():
                m.weight.fill_(1.0)
                m.bias.zero_()


def resnet18_nobn(pretrained=False, progress=True, **kwargs):
    model = ResNet_noBN(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet18'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model