#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
from torch.nn import functional

__author__ = 'Shayan Gharib, Konstantinos Drossos -- TUT'
__docformat__ = 'reStructuredText'
__all__ = ['LabelClassifier']


class LabelClassifier(nn.Module):
    """The label classifier.
    """

    def __init__(self, nb_output_classes):
        """Initialization of the label classifier.

        :param nb_output_classes: The number of classes to classify\
                                 (i.e. amount of outputs).
        :type nb_output_classes: int
        """
        super(LabelClassifier, self).__init__()

        self.linear_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.linear_layers.append(nn.Linear(in_features=1536, out_features=256))
        self.linear_layers.append(nn.Linear(in_features=256, out_features=256))
        for _ in range(2):
            self.dropouts.append(nn.Dropout(.25))

        self.output_layer = nn.Linear(in_features=256, out_features=nb_output_classes)

    def forward(self, x):
        """The forward pass of the label classifier.

        :param x: The input.
        :type x: torch.Tensor
        :return: The prediction of the label classifier.
        :rtype: torch.Tensor
        """
        output = x.view(x.size()[0], -1)
        for i in range(len(self.linear_layers)):
            output = self.dropouts[i](functional.relu(self.linear_layers[i](output)))

        return self.output_layer(output)

# EOF
