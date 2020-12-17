from networks.segnet import *
from networks.deeplab import *


def model_factory(model_name, num_input_bands, num_classes):
    if model_name == 'segnet':
        return SegNet(num_input_bands, num_classes)
    elif model_name == 'deeplab':
        return DeepLab(num_input_bands, num_classes)
    else:
        raise NotImplementedError('Network not identified: ' + model_name)
