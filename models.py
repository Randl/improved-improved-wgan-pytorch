from collections import OrderedDict

import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init


class WGAN(nn.Module):
    def __init__(self):
        super(WGAN, self).__init__()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal(m.weight, std=0.02)
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.02)
                if m.bias is not None:
                    init.constant(m.bias, 0)

    def _make_extra(self, layer_type, num_filters, n_extra_layers, drop=False, dropout=0):
        modules = OrderedDict()
        stage_name = "ExtraLayers"

        # Extra layers
        for i in range(n_extra_layers):
            name = stage_name + "_{}".format(i + 1)
            if drop:
                module = nn.Sequential(
                    layer_type(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters),
                    self.activation(inplace=True),
                    nn.Dropout(p=dropout, inplace=True))
            else:
                module = nn.Sequential(
                    layer_type(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters),
                    self.activation(inplace=True))
            modules[name] = module

        return nn.Sequential(modules)


class WGANGenerator(WGAN):
    def __init__(self, input_size=100, num_filters=64, out_channels=3, output_size=32, n_extra_layers=0,
                 activation=nn.ReLU):
        super(WGANGenerator, self).__init__()
        self.activation = activation
        self.has_extra = n_extra_layers > 0
        self.out_size = output_size

        first_filter = (self.out_size // 8) * num_filters
        self.first_conv_trans = nn.Sequential(
            nn.ConvTranspose2d(input_size, first_filter, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(first_filter),
            activation(inplace=True))

        self.conv_trans = self._make_conv_trans(first_filter)

        if self.has_extra:
            self.extra = self._make_extra(nn.ConvTranspose2d, num_filters, n_extra_layers)
        self.final_conv_trans = nn.Sequential(
            nn.ConvTranspose2d(num_filters, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh())

        self.init_params()

    def forward(self, x):
        x = self.first_conv_trans(x)
        x = self.conv_trans(x)
        if self.has_extra:
            x = self.extra(x)
        x = self.final_conv_trans(x)
        return x

    def _make_conv_trans(self, num_filters):
        modules = OrderedDict()
        stage_name = "ConvTranspose"

        # ConvTranspose layers
        for i in range(int(np.log2(self.out_size)) - 3):
            name = stage_name + "_{}".format(i + 1)
            module = nn.Sequential(
                nn.ConvTranspose2d(num_filters, num_filters // 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_filters // 2),
                self.activation(inplace=True))
            num_filters //= 2
            modules[name] = module

        return nn.Sequential(modules)


class WGANDiscriminator(WGAN):
    def __init__(self, input_size=32, in_channels=3, num_filters=64, n_extra_layers=0, activation=nn.LeakyReLU,
                 dropout_rate=0.5):
        super(WGANDiscriminator, self).__init__()
        self.activation = activation
        self.has_extra = n_extra_layers > 0
        self.input_size = input_size
        self.dropout_rate = dropout_rate

        self.first_conv = nn.Sequential(nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1),
                                        activation(inplace=True),
                                        nn.Dropout(p=dropout_rate, inplace=True))
        if self.has_extra:
            self.extra = self._make_extra(nn.Conv2d, num_filters, n_extra_layers, drop=True, dropout=dropout_rate)

        self.conv, last_filter = self._make_conv(num_filters)
        self.final_conv = nn.Conv2d(last_filter, 1, kernel_size=4, stride=1, padding=0, bias=False)

        self.drop  = False

        self.init_params()

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.training = not self.drop
        x = self.first_conv(x)
        if self.has_extra:
            x = self.extra(x)
        pre_last = self.conv(x)
        x = self.final_conv(pre_last)
        x = x.mean(0)
        return x, pre_last

    def _make_conv(self, num_filters):
        modules = OrderedDict()
        stage_name = "Conv"

        # Conv layers
        for i in range(int(np.log2(self.input_size)) - 3):
            name = stage_name + "_{}".format(i + 1)
            module = nn.Sequential(
                nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_filters * 2),
                self.activation(inplace=True),
                nn.Dropout(p=self.dropout_rate, inplace=True))
            num_filters *= 2
            modules[name] = module

        return nn.Sequential(modules), num_filters
