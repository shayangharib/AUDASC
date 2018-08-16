#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Shayan Gharib'
__all__ = ['device_exchange']


def device_exchange(data, device):
    """

    :param data: the data that needs to be transferred
    :param device: a string, the device that we want to use
    :return: transferred data e.g. from cpu to gpu
    """

    for key, value in data.items():
        data[key] = value.to(device)

    return data
