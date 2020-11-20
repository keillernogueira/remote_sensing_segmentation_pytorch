from networks.dynamic_dilated import *
from networks.fcn import *
from networks.pixelwise import *
from networks.segnet import *
from networks.unet import *
from networks.deeplab import deeplab


def model_factory(model_name, num_input_bands, num_classes):
    if model_name == 'dilated_grsl':
        return SegNet(num_input_bands, num_classes)
    else:
        raise NotImplementedError('Network not identified: ' + model_name)
