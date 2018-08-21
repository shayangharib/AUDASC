#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
from torch.nn import functional


class Discriminator(nn.Module):
    def __init__(self, nb_outputs):
        super(Discriminator, self).__init__()

        self.cnn_1 = nn.Conv2d(
            in_channels=128, out_channels=64,
            kernel_size=(3, 3), stride=1, padding=1
        )
        self.cnn_2 = nn.Conv2d(
            in_channels=64, out_channels=32,
            kernel_size=(3, 3), stride=1, padding=1
        )
        self.cnn_3 = nn.Conv2d(
            in_channels=32, out_channels=16,
            kernel_size=(3, 3), stride=1, padding=1
        )

        self.bn_1 = nn.BatchNorm2d(64)
        self.bn_2 = nn.BatchNorm2d(32)
        self.bn_3 = nn.BatchNorm2d(16)

        self.linear_1 = nn.Linear(in_features=192, out_features=nb_outputs)

    def forward(self, x):
        latent = self.bn_1(functional.relu(self.cnn_1(x)))
        latent = self.bn_2(functional.relu(self.cnn_2(latent)))
        latent = self.bn_3(functional.relu(self.cnn_3(latent)))
        output = latent.view(latent.size()[0], -1)

        return self.linear_1(output)
