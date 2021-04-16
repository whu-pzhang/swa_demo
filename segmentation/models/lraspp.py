import torch
import torch.nn as nn
from torch.nn import functional as F

import timm


class LRASPPHead(nn.Module):
    def __init__(self, low_channels, high_channels, num_classes,
                 inter_channels):
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, low, high):
        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = F.interpolate(x,
                          size=low.shape[-2:],
                          mode='bilinear',
                          align_corners=False)

        return self.low_classifier(low) + self.high_classifier(x)


class LRASPP(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        input_shape = x.size()[-2:]
        c2, c5 = self.backbone(x)
        output = self.head(c2, c5)

        output = F.interpolate(output,
                               size=input_shape,
                               mode='bilinear',
                               align_corners=False)

        return output


def lraspp_mobilenetv3_large(pretrained, in_channels, num_classes, **kwargs):
    backbone = timm.create_model('mobilenetv3_large_100',
                                 pretrained=True,
                                 in_chans=in_channels,
                                 features_only=True,
                                 output_stride=16,
                                 out_indices=[1, 4])
    c2, c5 = backbone.feature_info.channels()
    head = LRASPPHead(c2, c5, num_classes=num_classes, inter_channels=128)

    return LRASPP(backbone, head)
