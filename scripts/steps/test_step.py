#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

__author__ = 'Shayan Gharib -- TUT'
__docformat__ = 'reStructuredText'
__all__ = ['test_function', 'test_data_mini_batch']


def test_function(non_adapted_model, adapted_model, label_classifier, feat, labels):
    """

    :param non_adapted_model: non adapted cnn model
    :param adapted_model: adversarially adapted cnn
    :param label_classifier: label classifier
    :param feat: features of test data
    :param labels: labels of test data
    :return: the number of correct prediction by non adapted model,
             the number of correct prediction by adapted model,
             number of iterations
    """

    non_adapted_correct = []
    adapted_correct = []

    temp = 0

    for x, y in zip(feat, labels):

        x = x.float()
        y = torch.argmax(y, dim=1)

        source_mappings = non_adapted_model(x)
        non_adapted_y_hat = label_classifier(source_mappings)

        target_mappings = adapted_model(x)
        adapted_y_hat = label_classifier(target_mappings)

        non_adapted_y_hat = torch.argmax(non_adapted_y_hat, dim=1)
        adapted_y_hat = torch.argmax(adapted_y_hat, dim=1)

        non_adapted_correct.append((non_adapted_y_hat == y).float().sum())

        adapted_correct.append((adapted_y_hat == y).float().sum())

        temp += 1

    return non_adapted_correct, adapted_correct, temp


def test_data_mini_batch(feat, labels, batch_size):
    """

    :param feat: features of test data
    :param labels: labels of test data
    :param batch_size: mini batch size
    :return: mini batches of feature test data, corresponding labels of samples in each mini batch
    """
    mini_b_feat = [feat[i:i + batch_size] for i in range(0, len(feat), batch_size)]
    mini_b_labels = [labels[i:i + batch_size] for i in range(0, len(labels), batch_size)]

    return mini_b_feat, mini_b_labels

