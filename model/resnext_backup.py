import torch
import torch.nn as nn
from math import ceil

USE_MEMORY_EFFICIENT_SiLU = True

if USE_MEMORY_EFFICIENT_SiLU:
    @torch.jit.script
    def silu_fwd(x):
        return x.mul(torch.sigmoid(x))

    @torch.jit.script
    def silu_bwd(x, grad_output):
        x_sigmoid = torch.sigmoid(x)
        return grad_output * (x_sigmoid * (1. + x * (1. - x_sigmoid)))

    class SiLUJitImplementation(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return silu_fwd(x)

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            return silu_bwd(x, grad_output)

    def silu(x, inplace=False):
        return SiLUJitImplementation.apply(x)
else:
    def silu(x, inplace=False):
        return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())

class SiLU(nn.Module):
    def __init__(self, inplace=True):
        super(SiLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return silu(x, self.inplace)

def ConvBNSiLU(out, in_channels, channels, kernel=1, stride=1, pad=0, num_group=1):
    out.append(nn.Conv2d(in_channels, channels, kernel, stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(channels))
    out.append(SiLU(inplace=True))

class SE(nn.Module):
    def __init__(self, in_channels, channels, se_ratio=12):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, channels // se_ratio, kernel_size=1, padding=0),
            nn.BatchNorm2d(channels // se_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // se_ratio, channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class LinearBottleneck(nn.Module):
    def __init__(self, in_channels, channels, t, stride, use_se=True, se_ratio=12, dropout=0.0):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels <= channels
        self.in_channels = in_channels

        out = []
        if t != 1:
            dw_channels = in_channels * t
            ConvBNSiLU(out, in_channels, dw_channels)
        else:
            dw_channels = in_channels

        ConvBNSiLU(out, dw_channels, dw_channels, kernel=3, stride=stride, pad=1, num_group=dw_channels)
        if use_se:
            out.append(SE(dw_channels, dw_channels, se_ratio))

        ConvBNSiLU(out, dw_channels, channels)
        self.out = nn.Sequential(*out)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        out = self.out(x)
        if self.use_shortcut:
            out[:, :self.in_channels] += x
        return self.drop(out)

class ReXNetV1(nn.Module):
    def __init__(self, input_ch=3, final_ch=180, width_mult=1.0, depth_mult=1.0, classes=2,
                 use_se=True, se_ratio=12, dropout_ratio=0.2, dropout_path=0.25):
        super(ReXNetV1, self).__init__()

       
        layers = [1, 2, 2, 3]  
        strides = [1, 2, 2, 2]  
        use_ses = [False, False, True, True]  
        ts = [1] + [6] * (sum(layers) - 1)

        layers = [ceil(element * depth_mult) for element in layers]
        strides = sum([[s] + [1] * (layers[i] - 1) for i, s in enumerate(strides)], [])
        use_ses = sum([[use] * layers[i] for i, use in enumerate(use_ses)], [])

        self.depth = sum(layers) * 3
        stem_channel = 32 if width_mult >= 1.0 else 32 / width_mult

        inplanes = 96  
        features = []
        ConvBNSiLU(features, 3, inplanes, kernel=3, stride=2, pad=1)

        for block_idx in range(len(strides)):  
            out_channels = int(round((inplanes + final_ch / (self.depth // 3)) * width_mult))
            features.append(LinearBottleneck(inplanes, out_channels, t=ts[block_idx], stride=strides[block_idx], dropout=dropout_path, use_se=use_ses[block_idx]))
            inplanes = out_channels  

        pen_channels = 1280 
        ConvBNSiLU(features, inplanes, pen_channels)

        self.features = nn.Sequential(*features)
        self.out = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv2d(pen_channels, classes, 1, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.out(x).flatten(1)
        return x

