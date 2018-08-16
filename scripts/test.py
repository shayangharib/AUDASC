#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import path
import yaml

import torch

from aux import printing, file_io
from aux import math_funcs

from modules.model import Model
from modules.label_classifier import LabelClassifier

from scripts.steps import test_step

__author__ = 'Shayan Gharib -- TUT'


def testing(non_adapted_model_dir, adapted_model_dir, classifier_dir, nb_clss_labels,
            feat_path, labels_path, device, src_batch_size, trgt_batch_size):
    """Implements the complete test process of the AUDASC method

    :param non_adapted_model_dir: directory of non adapted model
    :param adapted_model_dir: directory of adapted model
    :param classifier_dir: directory of classifier
    :param nb_clss_labels: number of acoustic scene classes
    :param feat_path: directory of test features
    :param labels_path: directory of test labels
    :param device: The device that will be used.
    :param src_batch_size: source batch size
    :param trgt_batch_size: target batch size
    """
    non_adapted_cnn = Model().to(device)
    non_adapted_cnn.load_state_dict(torch.load(path.join(non_adapted_model_dir, 'non_adapted_cnn.pytorch')))

    adapted_cnn = Model().to(device)
    adapted_cnn.load_state_dict(torch.load(path.join(adapted_model_dir, 'target_cnn.pytorch')))

    label_classifier = LabelClassifier(nb_clss_labels).to(device)
    label_classifier.load_state_dict(torch.load(path.join(classifier_dir, 'label_classifier.pytorch')))

    non_adapted_cnn.train(False)
    adapted_cnn.train(False)
    label_classifier.train(False)

    feat = file_io.load_pickled_features(feat_path)
    labels = file_io.load_pickled_features(labels_path)

    non_adapted_acc = {}
    adapted_acc = {}

    '********************************************'
    '** testing for all data, device A, B, & C **'
    '********************************************'

    # testing on source data
    src_batch_feat, src_batch_labels = \
        test_step.test_data_mini_batch(feat['A'].to(device), labels['A'].to(device), batch_size=src_batch_size)
    non_adapted_src_correct, adapted_src_correct, src_temp = \
        test_step.test_function(non_adapted_cnn, adapted_cnn, label_classifier, src_batch_feat, src_batch_labels)

    non_adapted_src_len = src_temp * src_batch_size
    adapted_src_len = src_temp * src_batch_size

    # testing on target data
    target_feat = torch.cat([feat['B'], feat['C']], dim=0).to(device)
    target_labels = torch.cat([labels['B'], labels['C']], dim=0).to(device)

    trgt_batch_feat, trgt_batch_labels =\
        test_step.test_data_mini_batch(target_feat, target_labels, batch_size=trgt_batch_size)
    non_adapted_tgt_correct, adapted_tgt_correct, trgt_temp = \
        test_step.test_function(non_adapted_cnn, adapted_cnn, label_classifier, trgt_batch_feat, trgt_batch_labels)

    non_adapted_tgt_len = trgt_temp * trgt_batch_size
    adapted_tgt_len = trgt_temp * trgt_batch_size

    # calculating the accuracy of both models on data from device A
    non_adapted_acc['A'] = math_funcs.to_percentage(non_adapted_src_correct, non_adapted_src_len)
    adapted_acc['A'] = math_funcs.to_percentage(adapted_src_correct, adapted_src_len)

    # calculating the accuracy of both models on data from devices B & C
    non_adapted_acc['BC'] = math_funcs.to_percentage(non_adapted_tgt_correct, non_adapted_tgt_len)
    adapted_acc['BC'] = math_funcs.to_percentage(adapted_tgt_correct, adapted_tgt_len)

    # calculating the accuracy of both models on data from all devices
    non_adapted_beta, non_adapted_alpha = math_funcs.weighting_factors(non_adapted_src_len, non_adapted_tgt_len)
    adapted_beta, adapted_alpha = math_funcs.weighting_factors(adapted_src_len, adapted_tgt_len)

    non_adapted_weighted_acc = (non_adapted_beta * non_adapted_acc['A']) + (non_adapted_alpha * non_adapted_acc['BC'])
    adapted_weighted_acc = (adapted_beta * adapted_acc['A']) + (adapted_alpha * adapted_acc['BC'])

    non_adapted_acc['all'] = non_adapted_weighted_acc
    adapted_acc['all'] = adapted_weighted_acc

    printing.testing_result_msg(non_adapted_acc, adapted_acc, ending='\n', flushing=True)


def main():
    """Main function of the test script.

    Loading the non-/adapted models and classifier for testing
    process.
    """
    with open('learning_params.yml', 'r') as f:
        learner_params = yaml.load(f)

    device = 'cpu' if not torch.cuda.is_available() or not learner_params['general']['use_gpu'] else 'cuda'
    printing.info_msg('Process on {}'.format(device))

    nb_clss_labels = 10
    src_batch_size = 2
    target_batch_size = 36

    data_dir = str(learner_params['general']['data_dir'])
    non_adapted_model_dir = str(learner_params['test']['non_adapted_model_dir'])
    adapted_model_dir = str(learner_params['test']['adapted_model_dir'])
    classifier_dir = str(learner_params['test']['classifier_dir'])

    test_feat_path = path.join(data_dir, 'test_features.p')
    test_labels_path = path.join(data_dir, 'test_scene_labels.p')

    printing.info_msg('Test Phase')

    testing(non_adapted_model_dir, adapted_model_dir, classifier_dir,
            nb_clss_labels, test_feat_path, test_labels_path, device, src_batch_size, target_batch_size)


if __name__ == '__main__':
    main()
