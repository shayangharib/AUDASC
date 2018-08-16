#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import pickle

import torch

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = ['load_pickled_features', 'save_pytorch_model']


def load_pickled_features(file_name):
    """Loads a pickled file.

    :param file_name: The complete file name (i.e. full path with file name)
    :type file_name: str
    :return: The pickled file.
    :rtype: list | dict | torch.Tensor | object
    """
    kwargs = {'encoding': 'latin1'} if sys.version_info > (2, ) else {'protocol': 2}

    with open(file_name, 'rb') as f:
        return pickle.load(f, **kwargs)


def save_pytorch_model(the_model, the_file_name, the_directory, file_ext='pytorch'):
    """Saves a PyTorch model to disk.

    :param the_model: The model to be saved.
    :type the_model: torch.nn.Module | torch.optim.Optimizer
    :param the_file_name: The file name.
    :type the_file_name: str
    :param the_directory: The directory.
    :type the_directory: str
    :param file_ext: File extension
    :type file_ext: str
    """
    torch.save(
        the_model.state_dict(),
        os.path.join(the_directory, '{}.{}'.format(the_file_name, file_ext))
    )

# EOF
