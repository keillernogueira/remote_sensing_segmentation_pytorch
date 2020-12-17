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
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
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
        # print('unpooled ', torch.min(unpooled), torch.max(unpooled), torch.isnan(unpooled).any(), unpooled.size())
        return self.decode(unpooled)


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

        (enc1, ind1), size1 = self.enc1(x)  # size [batch, 64, 128, 128] # out [batch, 64, 64, 64]
        # print('enc1 ', torch.min(enc1), torch.max(enc1), torch.isnan(enc1).any(), torch.isnan(ind1).any(), enc1.size(), size1)
        (enc2, ind2), size2 = self.enc2(enc1)  # size [24, 128, 64, 64] # out [24, 128, 32, 32]
        # print('enc2 ', torch.min(enc2), torch.max(enc2), torch.isnan(enc2).any(), torch.isnan(ind2).any(), enc2.size(), size2)
        (enc3, ind3), size3 = self.enc3(enc2)  # size [24, 256, 32, 32] # out [24, 256, 16, 16]
        # print('enc3 ', torch.min(enc3), torch.max(enc3), torch.isnan(enc3).any(), torch.isnan(ind3).any(), enc3.size(), size3)

        dec3 = self.dec3(enc3, ind3, size3)  # out [24, 128, 32, 32]
        # print('dec3 ', torch.min(dec3), torch.max(dec3), torch.isnan(dec3).any(), dec3.size())
        dec2 = self.dec2(dec3, ind2, size2)  # out [24, 64, 64, 64]
        # print('dec2 ', torch.min(dec2), torch.max(dec2), torch.isnan(dec2).any(), dec2.size())
        dec1 = self.dec1(dec2, ind1, size1)  # out [24, 64, 128, 128]
        # print('dec1 ', torch.min(dec1), torch.max(dec1), torch.isnan(dec1).any(), dec1.size())

        final = self.final(dec1)
        # print('final ', torch.min(final), torch.max(final), torch.isnan(final).any(), final.size())

        if feat:
            return (final,
                    F.upsample(dec1, x.size()[2:], mode='bilinear'),
                    F.upsample(dec2, x.size()[2:], mode='bilinear'),
                    F.upsample(dec3, x.size()[2:], mode='bilinear'))
        else:
            return final


