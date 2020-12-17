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
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.encode(x)


class _ASPPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(_ASPPBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=atrous_rates[0], dilation=atrous_rates[0]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=atrous_rates[1], dilation=atrous_rates[1]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=atrous_rates[2], dilation=atrous_rates[2]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        _, _, h, w = x.shape
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)
        c4 = self.conv4(x)

        p1 = self.pool(x)
        c5 = self.conv5(p1)
        b1 = F.interpolate(c5, size=(h, w), mode="bilinear", align_corners=False)

        cat = torch.cat([c1, c2, c3, c4, b1], dim=1)
        return self.conv6(cat)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding):
        super(_DecoderBlock, self).__init__()

        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.decode(x)


class DeepLab(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(DeepLab, self).__init__()

        self.enc1 = _EncoderBlock(input_channels, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)

        self.aspp = _ASPPBlock(256, 256, atrous_rates=[2, 4, 8])

        self.low_level_features = _DecoderBlock(64, 256, kernel=1, padding=0)

        self.dec3 = _DecoderBlock(256*2, 256, kernel=3, padding=1)
        self.dec2 = _DecoderBlock(256, 256, kernel=3, padding=1)

        self.dec1 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)

        initialize_weights(self)

    def forward(self, x, feat=False):
        _, _, h, w = x.shape

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        aspp = self.aspp(enc3)

        low_level_features = self.low_level_features(enc1)

        _, _, h_lf, w_lf = low_level_features.shape
        b1 = F.interpolate(aspp, size=(h_lf, w_lf), mode="bilinear", align_corners=False)
        cat = torch.cat([b1, low_level_features], dim=1)

        dec3 = self.dec3(cat)
        dec2 = self.dec2(dec3)

        dec1 = self.dec1(dec2)

        final = F.interpolate(dec1, size=(h, w), mode="bilinear", align_corners=False)

        if feat:
            return (final,
                    F.upsample(dec1, x.size()[2:], mode='bilinear'),
                    F.upsample(dec2, x.size()[2:], mode='bilinear'))
        else:
            return final
