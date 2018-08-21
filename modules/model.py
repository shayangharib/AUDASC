#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
from torch.nn import functional


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.cnn_1 = nn.Conv2d(
            in_channels=1, out_channels=48,
            kernel_size=(11, 11), stride=(2, 3), padding=5
        )
        self.cnn_2 = nn.Conv2d(
            in_channels=48, out_channels=128,
            kernel_size=5, stride=(2, 3), padding=2
        )
        self.cnn_3 = nn.Conv2d(
            in_channels=128, out_channels=192,
            kernel_size=3, stride=1, padding=1
        )
        self.cnn_4 = nn.Conv2d(
            in_channels=192, out_channels=192,
            kernel_size=3, stride=1, padding=1
        )
        self.cnn_5 = nn.Conv2d(
            in_channels=192, out_channels=128,
            kernel_size=3, stride=1, padding=0
        )
        self.max_pool_1 = nn.MaxPool2d(
            kernel_size=3, stride=(1, 2)
        )
        self.max_pool_2 = nn.MaxPool2d(
            kernel_size=3, stride=(2, 2)
        )
        self.max_pool_3 = nn.MaxPool2d(
            kernel_size=3, stride=(1, 2)
        )

        self.bn_1 = nn.BatchNorm2d(48)
        self.bn_2 = nn.BatchNorm2d(128)
        self.bn_3 = nn.BatchNorm2d(128)

    def forward(self, x):
        output = self.bn_1(self.max_pool_1(functional.relu(self.cnn_1(x))))
        output = self.bn_2(self.max_pool_2(functional.relu(self.cnn_2(output))))

        output = functional.relu(self.cnn_3(output))
        output = functional.relu(self.cnn_4(output))
        output = self.bn_3(self.max_pool_3(functional.relu(self.cnn_5(output))))

        return output

# EOF
