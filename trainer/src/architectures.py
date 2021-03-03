from efficientnet_pytorch import EfficientNet


def efficientnet_b0(num_classes):
    override_params = {'num_classes': num_classes}
    return EfficientNet.from_name('efficientnet-b0', override_params=override_params)


def efficientnet_b3(num_classes):
    override_params = {'num_classes': num_classes}
    return EfficientNet.from_name('efficientnet-b3', override_params=override_params)
