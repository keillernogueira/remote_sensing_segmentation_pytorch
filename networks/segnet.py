import torch
import torch.nn.functional as F
from torch import nn
from networks.utils import initialize_weights


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_EncoderBlock, self).__init__()

        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        output = self.encode(x)
        return F.max_pool2d(output, 2, 2, return_indices=True), output.size()


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=2, dropout=False):
        super(_DecoderBlock, self).__init__()

        layers = [
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        ]

        if num_blocks == 3:
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(in_channels))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout2d()

        self.decode = nn.Sequential(*layers)

    def forward(self, x, indices, size):
        if self.dropout is not None:
            x = self.dropout(x)
        unpooled = F.max_unpool2d(x, indices, 2, 2, 0, size)
        return self.features(unpooled)


class SegNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SegNet, self).__init__()

        self.norm = torch.nn.LocalResponseNorm(2)

        self.enc1 = _EncoderBlock(input_channels, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)

        self.dec3 = _DecoderBlock(256, 128, num_blocks=3, dropout=True)
        self.dec2 = _DecoderBlock(128, 64, dropout=True)
        self.dec1 = _DecoderBlock(64, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

        initialize_weights(self)

    def forward(self, x, feat=False):
        x = self.norm(x)

        enc1, ind1, size1 = self.enc1(x)
        enc2, ind2, size2 = self.enc2(enc1)
        enc3, ind3, size3 = self.enc3(enc2)

        dec3 = self.dec3(enc3, ind3, size3)
        dec2 = self.dec2(dec3, ind2, size2)
        dec1 = self.dec1(dec2, ind1, size1)

        final = self.final(dec1)

        if feat:
            return (final,
                    F.upsample(dec1, x.size()[2:], mode='bilinear'),
                    F.upsample(dec2, x.size()[2:], mode='bilinear'),
                    F.upsample(dec3, x.size()[2:], mode='bilinear'))
        else:
            return final


