#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import makedirs, path
import yaml

import torch
from torch.nn import functional

from modules.model import Model
from modules.label_classifier import LabelClassifier
from modules.discriminator import Discriminator

from aux import printing, file_io
from aux.device_exchange import device_exchange

from scripts.steps import pre_training, domain_adaptation

__author__ = 'Shayan Gharib, Konstantinos Drossos -- TUT'
__docformat__ = 'reStructuredText'
__all__ = ['training_process']


def training_process(device, nb_class_labels, model_path, result_dir, patience, epochs, do_pre_train,
                     tr_feat_path, tr_labels_path, val_feat_path, val_labels_path,
                     tr_batch_size, val_batch_size,
                     adapt_patience, adapt_epochs, d_lr, tgt_lr, update_cnt, factor
                     ):
    """Implements the complete training process of the AUDASC method.

    :param device: The device that we will use.
    :type device: str
    :param nb_class_labels: The amount of labels for label classification.
    :type nb_class_labels: int
    :param model_path: The path of previously saved model (if any)
    :type model_path: str
    :param result_dir: The directory to save newly pre-trained model.
    :type result_dir: str
    :param patience: The patience for the pre-training step.
    :type patience: int
    :param epochs: The epochs for the pre-training step.
    :type epochs: int
    :param do_pre_train: Flag to indicate if we do pre-training.
    :type do_pre_train: bool
    :param tr_feat_path: The path for loading the training features.
    :type tr_feat_path: str
    :param tr_labels_path: The path for loading the training labels.
    :type tr_labels_path: str
    :param val_feat_path: The path for loading the validation features.
    :type val_feat_path: str
    :param val_labels_path: The path for loading the validation labels.
    :type val_labels_path: str
    :param tr_batch_size: The batch used for pre-training.
    :type tr_batch_size: int
    :param val_batch_size: The batch size used for validation.
    :type val_batch_size: int
    :param adapt_patience: The patience for the domain adaptation step.
    :type adapt_patience: int
    :param adapt_epochs: The epochs for the domain adaptation step.
    :type adapt_epochs: int
    :param d_lr: The learning rate for the discriminator.
    :type d_lr: float
    :param tgt_lr: The learning rate for the adapted model.
    :type tgt_lr: float
    :param update_cnt: An update controller for adversarial loss
    :type update_cnt: int
    :param factor: the coefficient used to be multiplied by classification loss.
    :type factor: int
    """

    tr_feat = device_exchange(file_io.load_pickled_features(tr_feat_path), device=device)
    tr_labels = device_exchange(file_io.load_pickled_features(tr_labels_path), device=device)
    val_feat = device_exchange(file_io.load_pickled_features(val_feat_path), device=device)
    val_labels = device_exchange(file_io.load_pickled_features(val_labels_path), device=device)

    loss_func = functional.cross_entropy

    non_adapted_cnn = Model().to(device)
    label_classifier = LabelClassifier(nb_class_labels).to(device)

    if not path.exists(result_dir):
        makedirs(result_dir)

    if do_pre_train:
        state_dict_path = result_dir

        printing.info_msg('Pre-training step')

        optimizer_source = torch.optim.Adam(
            list(non_adapted_cnn.parameters()) +
            list(label_classifier.parameters()), lr=1e-4
        )

        pre_training.pre_training(
            model=non_adapted_cnn, label_classifier=label_classifier, optimizer=optimizer_source,
            tr_batch_size=tr_batch_size, val_batch_size=val_batch_size,
            tr_feat=tr_feat['A'], tr_labels=tr_labels['A'], val_feat=val_feat['A'], val_labels=val_labels['A'],
            epochs=epochs, criterion=loss_func, patience=patience, result_dir=state_dict_path
        )

        del optimizer_source

    else:
        printing.info_msg('Loading a pre-trained non-adapted model')
        state_dict_path = model_path

    if not path.exists(state_dict_path):
        raise ValueError('The path for loading the pre trained model does not exist!')

    non_adapted_cnn.load_state_dict(
        torch.load(path.join(state_dict_path, 'non_adapted_cnn.pytorch'))
    )
    label_classifier.load_state_dict(
        torch.load(path.join(state_dict_path, 'label_classifier.pytorch'))
    )

    printing.info_msg('Training the Adversarial Adaptation Model')

    target_cnn = Model().to(device)
    target_cnn.load_state_dict(non_adapted_cnn.state_dict())
    discriminator = Discriminator(2).to(device)

    target_model_opt = torch.optim.Adam(target_cnn.parameters(), lr=tgt_lr)
    discriminator_opt = torch.optim.Adam(discriminator.parameters(), lr=d_lr)
    
    domain_adaptation.domain_adaptation(
        non_adapted_cnn, target_cnn, label_classifier, discriminator,
        target_model_opt, discriminator_opt,
        loss_func, loss_func, loss_func,
        tr_feat, tr_labels, val_feat, val_labels, adapt_epochs,
        update_cnt, result_dir, adapt_patience, device, factor
    )
    
    
def main():
    """Main function of the training script.

    Basic bootstrapping and initializations for initiating the training\
    process.
    """
    with open('learning_params.yml', 'r') as f:
        learner_params = yaml.load(f)

    device = 'cpu' if not torch.cuda.is_available() or not learner_params['general']['use_gpu'] else 'cuda'

    printing.info_msg('Process on {}'.format(device))

    data_dir = str(learner_params['general']['data_dir'])
    result_dir = str(learner_params['general']['result_dir'])
    patience = int(learner_params['general']['patience'])
    epochs = int(learner_params['general']['epochs'])
    do_pre_train = learner_params['general']['pre_training']
    tr_batch_size = int(learner_params['general']['tr_batch_size'])
    val_batch_size = int(learner_params['general']['val_batch_size'])

    nb_class_labels = 10

    tr_feat_path = path.join(data_dir, 'training_features.p')
    tr_labels_path = path.join(data_dir, 'training_scene_labels.p')
    val_feat_path = path.join(data_dir, 'validation_features.p')
    val_labels_path = path.join(data_dir, 'validation_scene_labels.p')
    model_path = learner_params['general']['saved_model_dir']

    adapt_patience = learner_params['adaptation']['patience']
    adapt_epochs = learner_params['adaptation']['epochs']
    d_lr = float(learner_params['adaptation']['d_lr'])
    tgt_lr = float(learner_params['adaptation']['tgt_lr'])
    update_cnt = int(learner_params['adaptation']['update_controller'])
    factor = int(learner_params['adaptation']['cls_loss_w'])

    training_process(
        device=device, nb_class_labels=nb_class_labels,
        model_path=model_path, result_dir=result_dir,
        patience=patience, epochs=epochs, do_pre_train=do_pre_train,
        tr_feat_path=tr_feat_path, tr_labels_path=tr_labels_path,
        val_feat_path=val_feat_path, val_labels_path=val_labels_path,
        tr_batch_size=tr_batch_size, val_batch_size=val_batch_size,
        adapt_patience=adapt_patience, adapt_epochs=adapt_epochs,
        d_lr=d_lr, tgt_lr=tgt_lr, update_cnt=update_cnt, factor=factor
    )


if __name__ == '__main__':
    main()

# EOF
