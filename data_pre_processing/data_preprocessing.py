#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import glob
import yaml
import pickle
import random
from collections import defaultdict

import pandas as pd
import csv

import os
from os import path

from aux import printing, file_io

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit

import torch

__author__ = 'Shayan Gharib -- TUT'


def normalizer(feature_matrix, mu, sd):
    """

    :param feature_matrix: extracted features
    :param mu: mean values of each band
    :param sd: standard deviation
    :return: normalized features
    """
    feature_matrix = (feature_matrix - mu) / sd

    return feature_matrix


def get_data(train_feat, train_cls_labels, train_device_labels, test_feat, test_cls_labels, test_device_labels,
             mean, std, channel, axis, normalize):
    """
    :param train_feat: training features.
    :param train_cls_labels: acoustic scene labels for training features.
    :param train_device_labels: domain labels for training features.
    :param test_feat: features for test data.
    :param test_cls_labels: acoustic scene labels for test features
    :param test_device_labels: domain labels for test features
    :param mean: mean values of each band
    :param std: standard deviation
    :param channel: extending the dimension of data if used as input to CNN
    :param axis: adding the channel to first or last axis of data
    :param normalize: boolean parameter to allow normalization process on data
    
    :return: splitting features in training, validation, test together with corresponding labels  
    """

    # source- device A
    source_train_index = [i for i, e in enumerate(train_device_labels) if e == 'a']
    source_test_index = [i for i, e in enumerate(test_device_labels) if e == 'a']

    src_tr_cls_labels = [train_cls_labels[i] for i in source_train_index]
    src_test_cls_labels = [test_cls_labels[i] for i in source_test_index]

    src_tr_feat = train_feat[source_train_index, :, :]
    src_test_feat = test_feat[source_test_index, :, :]

    # target- device B
    device_b_tr_index = [i for i, e in enumerate(train_device_labels) if e == 'b']
    device_b_test_index = [i for i, e in enumerate(test_device_labels) if e == 'b']

    device_b_tr_cls_labels = [train_cls_labels[i] for i in device_b_tr_index]
    device_b_test_cls_labels = [test_cls_labels[i] for i in device_b_test_index]

    device_b_tr_feat = train_feat[device_b_tr_index, :, :]
    device_b_test_feat = test_feat[device_b_test_index, :, :]

    # target- device C
    device_c_tr_index = [i for i, e in enumerate(train_device_labels) if e == 'c']
    device_c_test_index = [i for i, e in enumerate(test_device_labels) if e == 'c']

    device_c_tr_cls_labels = [train_cls_labels[i] for i in device_c_tr_index]
    device_c_test_cls_labels = [test_cls_labels[i] for i in device_c_test_index]

    device_c_tr_feat = train_feat[device_c_tr_index, :, :]
    device_c_test_feat = test_feat[device_c_test_index, :, :]

    # Encode labels with value between 0 and n_classes-1 "words --> integers"
    class_label_enc = LabelEncoder()
    class_label_enc.fit(train_cls_labels)

    src_tr_cls_labels = class_label_enc.transform(src_tr_cls_labels)
    src_test_cls_labels = class_label_enc.transform(src_test_cls_labels)

    device_b_tr_cls_labels = class_label_enc.transform(device_b_tr_cls_labels)
    device_b_test_cls_labels = class_label_enc.transform(device_b_test_cls_labels)

    device_c_tr_cls_labels = class_label_enc.transform(device_c_tr_cls_labels)
    device_c_test_cls_labels = class_label_enc.transform(device_c_test_cls_labels)

    # Encode integer labels to binary values(OneHot encoded)
    scene_hot_enc = OneHotEncoder()

    src_tr_cls_labels = scene_hot_enc.fit_transform(np.expand_dims(src_tr_cls_labels, axis=-1)).toarray()
    src_test_cls_labels = scene_hot_enc.transform(np.expand_dims(src_test_cls_labels, axis=-1)).toarray()

    device_b_tr_cls_labels = scene_hot_enc.transform(np.expand_dims(device_b_tr_cls_labels, axis=-1)).toarray()
    device_b_test_cls_labels = scene_hot_enc.transform(np.expand_dims(device_b_test_cls_labels, axis=-1)).toarray()

    device_c_tr_cls_labels = scene_hot_enc.transform(np.expand_dims(device_c_tr_cls_labels, axis=-1)).toarray()
    device_c_test_cls_labels = scene_hot_enc.transform(np.expand_dims(device_c_test_cls_labels, axis=-1)).toarray()

    # normalizing the data
    if normalize:
        src_tr_feat = normalizer(src_tr_feat, mean, std)
        src_test_feat = normalizer(src_test_feat, mean, std)

        device_b_tr_feat = normalizer(device_b_tr_feat, mean, std)
        device_b_test_feat = normalizer(device_b_test_feat, mean, std)

        device_c_tr_feat = normalizer(device_c_tr_feat, mean, std)
        device_c_test_feat = normalizer(device_c_test_feat, mean, std)

    # adding channel to the data to be suited as input to CNN
    if channel:
        if axis == 'first':
            src_tr_feat = np.expand_dims(src_tr_feat, axis=1)
            src_test_feat = np.expand_dims(src_test_feat, axis=1)

            device_b_tr_feat = np.expand_dims(device_b_tr_feat, axis=1)
            device_b_test_feat = np.expand_dims(device_b_test_feat, axis=1)

            device_c_tr_feat = np.expand_dims(device_c_tr_feat, axis=1)
            device_c_test_feat = np.expand_dims(device_c_test_feat, axis=1)

        elif axis == 'last':
            src_tr_feat = np.expand_dims(src_tr_feat, axis=-1)
            src_test_feat = np.expand_dims(src_test_feat, axis=-1)

            device_b_tr_feat = np.expand_dims(device_b_tr_feat, axis=-1)
            device_b_test_feat = np.expand_dims(device_b_test_feat, axis=-1)

            device_c_tr_feat = np.expand_dims(device_c_tr_feat, axis=-1)
            device_c_test_feat = np.expand_dims(device_c_test_feat, axis=-1)

    src_sss = StratifiedShuffleSplit(n_splits=1, test_size=612, random_state=2018)
    for train_index, test_index in src_sss.split(src_tr_feat, src_tr_cls_labels):
        src_tr_feat, src_val_feat = src_tr_feat[train_index], src_tr_feat[test_index]
        src_tr_cls_labels, src_val_cls_labels = src_tr_cls_labels[train_index], src_tr_cls_labels[test_index]

    b_sss = StratifiedShuffleSplit(n_splits=1, test_size=54, random_state=2018)
    for train_index, test_index in b_sss.split(device_b_tr_feat, device_b_tr_cls_labels):
        device_b_tr_feat, device_b_val_feat = device_b_tr_feat[train_index], device_b_tr_feat[test_index]
        device_b_tr_cls_labels, device_b_val_cls_labels = device_b_tr_cls_labels[train_index], device_b_tr_cls_labels[test_index]

    c_sss = StratifiedShuffleSplit(n_splits=1, test_size=54, random_state=2018)
    for train_index, test_index in c_sss.split(device_c_tr_feat, device_c_tr_cls_labels):
        device_c_tr_feat, device_c_val_feat = device_c_tr_feat[train_index], device_c_tr_feat[test_index]
        device_c_tr_cls_labels, device_c_val_cls_labels = device_c_tr_cls_labels[train_index], device_c_tr_cls_labels[test_index]

    title_a = 'The shape of the source features- device A:'
    printing.get_data_msg(src_tr_feat, src_val_feat, src_test_feat, title_a, ending='\n', flushing=True)

    title_b = 'The shape of the target features- device B:'
    printing.get_data_msg(device_b_tr_feat, device_b_val_feat, device_b_test_feat, title_b,
                          ending='\n', flushing=True)

    title_c = 'The shape of the target features- device C:'
    printing.get_data_msg(device_c_tr_feat, device_c_val_feat, device_c_test_feat, title_c,
                          ending='\n', flushing=True)

    tr_feat = {}
    tr_labels = {}
    val_feat = {}
    val_labels = {}
    test_feat = {}
    test_labels = {}

    tr_feat['A'] = torch.from_numpy(src_tr_feat)
    tr_feat['B'] = torch.from_numpy(device_b_tr_feat)
    tr_feat['C'] = torch.from_numpy(device_c_tr_feat)

    tr_labels['A'] = torch.from_numpy(src_tr_cls_labels)
    tr_labels['B'] = torch.from_numpy(device_b_tr_cls_labels)
    tr_labels['C'] = torch.from_numpy(device_c_tr_cls_labels)

    val_feat['A'] = torch.from_numpy(src_val_feat)
    val_feat['B'] = torch.from_numpy(device_b_val_feat)
    val_feat['C'] = torch.from_numpy(device_c_val_feat)

    val_labels['A'] = torch.from_numpy(src_val_cls_labels)
    val_labels['B'] = torch.from_numpy(device_b_val_cls_labels)
    val_labels['C'] = torch.from_numpy(device_c_val_cls_labels)

    test_feat['A'] = torch.from_numpy(src_test_feat)
    test_feat['B'] = torch.from_numpy(device_b_test_feat)
    test_feat['C'] = torch.from_numpy(device_c_test_feat)

    test_labels['A'] = torch.from_numpy(src_test_cls_labels)
    test_labels['B'] = torch.from_numpy(device_b_test_cls_labels)
    test_labels['C'] = torch.from_numpy(device_c_test_cls_labels)

    return tr_feat, tr_labels, val_feat, val_labels, test_feat, test_labels


def main():

    """Main function of the data_preprocessing script.

    Loading the audio files and extracting the mel band features.
    """

    with open('feature_params.yml', 'r') as f:
        feature_params = yaml.load(f)

    channel = feature_params['preprocessing']['channel']
    axis = str(feature_params['preprocessing']['axis'])
    normalize = feature_params['preprocessing']['normalize']
    n_time_frames = int(feature_params['preprocessing']['n_time_frames'])
    n_mel = feature_params['preprocessing']['n_mel_bands']

    feature_set_dir = feature_params['general']['feature_set_dir']
    test_setup = feature_params['general']['test_setup']
    meta = feature_params['general']['meta_file_dir']

    ready_to_use_feat_dir = feature_params['general']['ready_to_use_feat_dir']
    if not os.path.exists(ready_to_use_feat_dir):
        os.makedirs(ready_to_use_feat_dir)

    file_list = sorted(glob.glob(os.path.join(feature_set_dir, '*.p')))
    feature_based_meta = [i.split('/')[-1].replace('.p', '.wav') for i in file_list]

    csv_reader = csv.reader(open(test_setup, 'r'), delimiter='\t')
    test_list = []
    for row in csv_reader:
        test_list.append(row[0].replace('audio/', ''))

    meta_data = pd.read_csv(meta, sep='\t')
    
    labels_list = defaultdict(dict)
    for i in range(len(meta_data)):
        labels_list[meta_data.loc[i].filename.replace('audio/', '')]['scene_label'] = meta_data.loc[i].scene_label
        labels_list[meta_data.loc[i].filename.replace('audio/', '')]['identifier'] = meta_data.loc[i].identifier
        labels_list[meta_data.loc[i].filename.replace('audio/', '')]['source_label'] = meta_data.loc[i].source_label

    assert len(meta_data) == len(feature_based_meta), 'the number of features is not the ' \
                                                      'same as the length of meta data'
    val_samples = len(test_list)
    train_samples = len(meta_data) - val_samples

    train_feat = np.zeros((train_samples, n_mel, n_time_frames))
    val_feat = np.zeros((val_samples, n_mel, n_time_frames))

    train_scene_labels_name = []
    val_scene_labels_name = []

    train_device_labels_name = []
    val_device_labels_name = []
    
    # building the training list by excluding the validation files from meta data
    training_list = sorted(list(set(feature_based_meta) - set(test_list)))

    # Double check that training and validation files are not overlapping.
    set_contamination = set(training_list).intersection(test_list)
    if set_contamination:
        raise ValueError('Training and validation file lists are overlapping!')

    t, v = 0, 0
    mean_list = []
    std_list = []

    for ind, item in enumerate(file_list):
        
        if item.split('/')[-1].replace('.p', '.wav') in test_list:
            val_feat[v, :, :] = pickle.load(open(item, 'rb'))['feat'][0]
            val_scene_labels_name.append(labels_list[item.split('/')[-1].replace('.p', '.wav')]['scene_label'])
            val_device_labels_name.append(labels_list[item.split('/')[-1].replace('.p', '.wav')]['source_label'])
            v += 1
            
        elif item.split('/')[-1].replace('.p', '.wav') in training_list:
            tmp = pickle.load(open(item, 'rb'))
            train_feat[t, :, :] = tmp['feat'][0]
            mean_list.append((np.expand_dims(tmp['stat'][0]['mean'], axis=1)))
            std_list.append((np.expand_dims(tmp['stat'][0]['std'], axis=1)))
            train_scene_labels_name.append(labels_list[item.split('/')[-1].replace('.p', '.wav')]['scene_label'])
            train_device_labels_name.append(labels_list[item.split('/')[-1].replace('.p', '.wav')]['source_label'])
            t += 1

    # preparing the mean and std of training data to be used for normalizing the whole dataset
    mean = np.mean(np.concatenate(mean_list, axis=1), axis=1, keepdims=True)
    std = np.mean(np.concatenate(std_list, axis=1), axis=1, keepdims=True)

    # preparing the features as input to DNN model
    tr_feat, tr_labels, val_feat, val_labels, test_feat, test_labels = \
        get_data(train_feat, train_scene_labels_name, train_device_labels_name,
                 val_feat, val_scene_labels_name, val_device_labels_name,
                 mean, std, channel, axis, normalize)

    pickle.dump(tr_feat, open(os.path.join(ready_to_use_feat_dir, "training_features.p"), "wb"), protocol=2)
    pickle.dump(tr_labels, open(os.path.join(ready_to_use_feat_dir, "training_scene_labels.p"), "wb"), protocol=2)
    pickle.dump(val_feat, open(os.path.join(ready_to_use_feat_dir, "validation_features.p"), "wb"), protocol=2)
    pickle.dump(val_labels, open(os.path.join(ready_to_use_feat_dir, "validation_scene_labels.p"), "wb"), protocol=2)
    pickle.dump(test_feat, open(os.path.join(ready_to_use_feat_dir, "test_features.p"), "wb"), protocol=2)
    pickle.dump(test_labels, open(os.path.join(ready_to_use_feat_dir, "test_scene_labels.p"), "wb"), protocol=2)


if __name__ == "__main__":
    main()
