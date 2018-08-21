#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
from torch.nn import functional


class LabelClassifier(nn.Module):

    def __init__(self, nb_output_classes):
        super(LabelClassifier, self).__init__()

        self.linear_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.linear_layers.append(nn.Linear(in_features=1536, out_features=256))
        self.linear_layers.append(nn.Linear(in_features=256, out_features=256))
        for _ in range(2):
            self.dropouts.append(nn.Dropout(.25))

        self.output_layer = nn.Linear(in_features=256, out_features=nb_output_classes)

    def forward(self, x):
        output = x.view(x.size()[0], -1)
        for i in range(len(self.linear_layers)):
            output = self.dropouts[i](functional.relu(self.linear_layers[i](output)))

        return self.output_layer(output)

# EOF
