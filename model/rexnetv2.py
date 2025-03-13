import torch
import torch.nn as nn
from math import ceil

class SiLU(nn.Module):
    def __init__(self, inplace=True):
        super(SiLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul(torch.sigmoid(x))

def ConvBNSiLU(out, in_channels, channels, kernel=1, stride=1, pad=0, num_group=1):
    out.append(nn.Conv2d(in_channels, channels, kernel, stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(channels))
    out.append(SiLU(inplace=True))

class LinearBottleneck(nn.Module):
    def __init__(self, in_channels, channels, t, stride, use_channel_attn=True, channel_attn_param=12, drop_path=0.0):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels <= channels
        self.in_channels = in_channels

        out = []
        if t != 1:
            dw_channels = in_channels * t
            ConvBNSiLU(out, in_channels, dw_channels)

        ConvBNSiLU(out, dw_channels, channels)
        self.out = nn.Sequential(*out)
        self.drop = nn.Dropout2d(drop_path)

    def forward(self, x):
        out = self.out(x)
        if self.use_shortcut:
            out[:, :self.in_channels] += x
        return self.drop(out)

class ReXNetV2(nn.Module):
    def __init__(self, input_ch=16, final_ch=180, width_mult=1.0, depth_mult=1.0, classes=2,
                 use_channel_attn=True, channel_attn_param=3, dropout_ratio=0.2, drop_path=0.25):
        super(ReXNetV2, self).__init__()

        layers = [1, 2, 2, 3, 3, 5]
        strides = [1, 2, 2, 2, 1, 2]
        use_channel_attns = [False, False, True, True, True, True]

        layers = [ceil(element * depth_mult) for element in layers]
        strides = sum([[s] + [1] * (layers[i] - 1) for i, s in enumerate(strides)], [])
        use_channel_attns = sum([[use] * layers[i] for i, use in enumerate(use_channel_attns)], [])

        ts = [1] * layers[0] + [6] * sum(layers[1:])

        self.depth = sum(layers) * 3
        stem_channel = 32 if width_mult >= 1.0 else 32 / width_mult
        inplanes = input_ch if width_mult >= 1.0 else input_ch / width_mult
        #inplanes = input_ch
        features = []
        ConvBNSiLU(features, 3, int(stem_channel * width_mult), kernel=3, stride=2, pad=1)

        for i in range(self.depth // 3):
            features.append(LinearBottleneck(inplanes, int(stem_channel * width_mult), t=6, stride=1, drop_path=drop_path))

        pen_channels = int(1280 * width_mult)
        ConvBNSiLU(features, stem_channel, pen_channels)

        self.features = nn.Sequential(*features)
        self.out = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv2d(pen_channels, classes, 1, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.out(x).flatten(1)
        return x

