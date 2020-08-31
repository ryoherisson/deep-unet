""" Full assembly of the parts to form the complete network 

Reference
    Original author: milesial
    https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
"""

import torch.nn as nn

from modeling.deep_unet.modules.unet_parts import DoubleConv, Down, Up, OutConv


class DeepUNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_filters_start=32, growth_factor=2, bilinear=True):
        super(DeepUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        n_filters = n_filters_start # 32

        self.inc = DoubleConv(n_channels, n_filters)
        self.down1 = Down(n_filters, n_filters*growth_factor)
        n_filters *= growth_factor # 64
        self.down2 = Down(n_filters, n_filters*growth_factor)
        n_filters *= growth_factor # 128
        self.down3 = Down(n_filters, n_filters*growth_factor)
        n_filters *= growth_factor # 256
        self.down4 = Down(n_filters, n_filters*growth_factor)
        n_filters *= growth_factor # 512

        factor = 2 if bilinear else 1
        self.down5 = Down(n_filters, n_filters*growth_factor//factor)

        n_filters *= growth_factor # 1024

        self.up1 = Up(n_filters, n_filters//growth_factor//factor, bilinear)
        n_filters //= growth_factor # 512
        self.up2 = Up(n_filters, n_filters//growth_factor//factor, bilinear)
        n_filters //= growth_factor # 256
        self.up3 = Up(n_filters, n_filters//growth_factor//factor, bilinear)
        n_filters //= growth_factor # 128
        self.up4 = Up(n_filters, n_filters//growth_factor//factor, bilinear)
        n_filters //= growth_factor # 64
        self.up5 = Up(n_filters, n_filters//growth_factor, bilinear)
        n_filters //= growth_factor # 32
        self.outc = OutConv(n_filters, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        logits = self.outc(x)
        return logits