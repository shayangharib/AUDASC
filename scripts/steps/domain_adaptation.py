#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import shuffle
from itertools import cycle

import torch

from aux import printing, math_funcs, file_io

from modules.plot import plot_function

__author__ = 'Shayan Gharib, Konstantinos Drossos -- TUT'
__docformat__ = 'reStructuredText'
__all__ = ['domain_adaptation']


def _forward_pass(x, y_label, y_domain,
                  source_model, target_model, label_classifier, discriminator,
                  target_criterion, cls_criterion, adversarial_criterion,
                  src_batch_size):
    """Performs one forward pass at the domain adaptation step.

    :param x: The input data.
    :type x: torch.Tensor
    :param y_label: The target label data.
    :type y_label: torch.Tensor
    :param y_domain: The target domain data.
    :type y_domain: torch.Tensor
    :param source_model: The source model.
    :type source_model: torch.nn.Module
    :param target_model: The target model.
    :type target_model: torch.nn.Module
    :param label_classifier: The label classifier.
    :type label_classifier: torch.nn.Module
    :param discriminator: The discriminator.
    :type discriminator: torch.nn.Module
    :param target_criterion: The criterion for the target model.
    :type target_criterion: callable
    :param cls_criterion: The criterion for the label classifier.
    :type cls_criterion: callable
    :param adversarial_criterion: The criterion for the domain classifier.
    :type adversarial_criterion: callable
    :param src_batch_size: The batch size of the source domain data.
    :type src_batch_size: int
    :return: The three losses: mapping, label classification, and domain classification, \
             plus the accuracy for the labels of the source and target domains.
    :rtype: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
    """
    source_out = source_model(x[:src_batch_size])
    target_out = target_model(x)

    cls_y_hat = label_classifier(target_out)
    y_hat = discriminator(torch.cat([source_out, target_out[src_batch_size:]]))

    adversarial_loss = adversarial_criterion(y_hat, y_domain)
    mapping_loss = target_criterion(y_hat[src_batch_size:], 1 - y_domain[src_batch_size:])
    cls_loss = cls_criterion(cls_y_hat[:src_batch_size], y_label[:src_batch_size])

    cls_y_hat = torch.argmax(cls_y_hat, dim=1)

    acc_src = (cls_y_hat[:src_batch_size] == y_label[:src_batch_size]).float().sum()
    acc_trgt = (cls_y_hat[src_batch_size:] == y_label[src_batch_size:]).float().sum()

    return mapping_loss, cls_loss, adversarial_loss, acc_src, acc_trgt


def _get_mini_batches(x, y, src_b_size, target_b_size, device, shuffle_data=True):
    """Iterator for providing the mini-batch data for the domain adaptation step.

    :param x: The input data.
    :type x: dict[str, torch.Tensor]
    :param y: The target labels.
    :type y: dict[str, torch.Tensor]
    :param src_b_size: The batch size for the source domain data.
    :type src_b_size: int
    :param target_b_size: The batch size for the target domain data.
    :type target_b_size: int
    :param device: the device that will be used.
    :type device: string
    :param shuffle_data: Shall we shuffle the data?
    :type shuffle_data: bool
    :return: The mini-batch of input, labels, and domain.
    :rtype: (torch.Tensor, torch.Tensor, torch.Tensor)
    """
    src_tr_indices = list(range(len(x['A'])))
    b_tr_indices = list(range(len(x['B'])))
    c_tr_indices = list(range(len(x['C'])))

    if shuffle_data:
        shuffle(src_tr_indices)
        shuffle(b_tr_indices)
        shuffle(c_tr_indices)

    src_len_it = range(0, len(x['A']), src_b_size)
    target_len = min(len(x['B']), len(x['C']))
    target_it = range(0, target_len, target_b_size)

    for i_scr, i_target in zip(src_len_it, cycle(target_it)):
        input_x = torch.cat((
            x['A'][src_tr_indices[i_scr:i_scr + src_b_size]],
            x['B'][b_tr_indices[i_target:i_target + target_b_size]],
            x['C'][c_tr_indices[i_target:i_target + target_b_size]]
        )).float()

        label_y = torch.cat((
            y['A'][src_tr_indices[i_scr:i_scr + src_b_size]],
            y['B'][b_tr_indices[i_target:i_target + target_b_size]],
            y['C'][c_tr_indices[i_target:i_target + target_b_size]]
        )).argmax(dim=1).long()

        y_discriminator = torch.ones(src_b_size + (2 * target_b_size))
        y_discriminator[src_b_size:] = 0

        yield input_x, label_y, y_discriminator.to(device).long()


def _training(tr_feat, tr_labels,
              source_model, target_model, label_classifier, discriminator,
              target_criterion, cls_criterion, adversarial_criterion,
              target_model_opt, discriminator_opt,
              src_batch_size, device_batch_size, update_cnt, device, factor):
    """The domain adaptation training process.

    :param tr_feat: The training input features.
    :type tr_feat: dict[str, torch.Tensor]
    :param tr_labels: The training targeted values
    :type tr_labels: dict[str, torch.Tensor]
    :param source_model: The non-adapted source model.
    :type source_model: torch.nnn.Module
    :param target_model: The target model, to be adapted.
    :type target_model: torch.nn.Module
    :param label_classifier: The pre-trained label classifier.
    :type label_classifier: torch.nn.Module
    :param discriminator: The domain discriminator.
    :type discriminator: torch.nn.Module
    :param target_criterion: The criterion for the target model.
    :type target_criterion: callable
    :param cls_criterion: The criterion for the label classifier.
    :type cls_criterion: callable
    :param adversarial_criterion: The adversarial criterion.
    :type adversarial_criterion: callable
    :param target_model_opt: The optimizer of the target model.
    :type target_model_opt: torch.optim.Optimizer
    :param discriminator_opt: The optimizer for the discriminator.
    :type discriminator_opt: torch.optim.Optimizer
    :param src_batch_size: The batch size for the source domain data.
    :type src_batch_size: int
    :param device_batch_size: The batch size for the target domain data.
    :type device_batch_size: int
    :param update_cnt: The update counter for updating the parameters of the\
                       discriminator.
    :type update_cnt: int
    :param device: the device that will be used.
    :type device: string
    :param factor: the coefficient used to be multiplied by classification loss.
    :type factor: int
    :return: The average loss of one epoch for mappings, label classification,\
             and domain classification, the average accuracy for the source\
             and the target models, and the average weighted accuracy.
    :rtype: (float, float, float, float, float, float)
    """
    target_model.train(True)
    discriminator.train(True)

    source_iterations_acc = []
    target_iterations_acc = []

    mapping_iterations_loss = []
    cls_iterations_loss = []
    domain_iterations_loss = []

    temp = 1

    for x, cls_y, d_y in _get_mini_batches(
            x=tr_feat, y=tr_labels, src_b_size=src_batch_size,
            target_b_size=device_batch_size, device=device):

        discriminator_opt.zero_grad()
        target_model_opt.zero_grad()

        mapping_loss, cls_loss, adversarial_loss, acc_src, acc_trgt = _forward_pass(
            x=x, y_label=cls_y, y_domain=d_y,
            source_model=source_model, target_model=target_model,
            label_classifier=label_classifier, discriminator=discriminator,
            target_criterion=target_criterion, cls_criterion=cls_criterion,
            adversarial_criterion=adversarial_criterion, src_batch_size=src_batch_size
        )

        if divmod(temp, update_cnt)[1] == 0:
            adversarial_loss.backward(retain_graph=True)
            discriminator_opt.step()
            discriminator_opt.zero_grad()

        total_loss = mapping_loss + cls_loss.mul(factor)
        total_loss.backward()

        target_model_opt.step()
        target_model_opt.zero_grad()

        mapping_iterations_loss.append(mapping_loss.item())
        cls_iterations_loss.append(cls_loss.item())
        domain_iterations_loss.append(adversarial_loss.item())
        
        source_iterations_acc.append(acc_src)
        target_iterations_acc.append(acc_trgt)

        temp += 1

    mapping_epoch_loss = math_funcs.avg(mapping_iterations_loss)
    cls_epoch_loss = math_funcs.avg(cls_iterations_loss)
    domain_epoch_loss = math_funcs.avg(domain_iterations_loss)

    nb_src = (temp - 1) * src_batch_size
    nb_trg = (temp - 1) * 2 * device_batch_size

    source_epoch_acc = math_funcs.to_percentage(source_iterations_acc, nb_src)
    target_epoch_acc = math_funcs.to_percentage(target_iterations_acc, nb_trg)

    w_src, w_trgt = math_funcs.weighting_factors(nb_src, nb_trg)

    w_acc = w_src * source_epoch_acc + w_trgt * target_epoch_acc

    return mapping_epoch_loss, cls_epoch_loss, domain_epoch_loss, source_epoch_acc, target_epoch_acc, w_acc


def _validation(val_feat, val_labels,
                source_model, target_model, label_classifier, discriminator,
                target_criterion, cls_criterion, adversarial_criterion,
                src_batch_size, device_batch_size, device):
    """The domain adaptation validation process.

    :param val_feat: The validation input features.
    :type val_feat: dict[str, torch.Tensor]
    :param val_labels: The validation targeted values
    :type val_labels: dict[str, torch.Tensor]
    :param source_model: The non-adapted source model.
    :type source_model: torch.nnn.Module
    :param target_model: The target model, to be adapted.
    :type target_model: torch.nn.Module
    :param label_classifier: The pre-trained label classifier.
    :type label_classifier: torch.nn.Module
    :param discriminator: The domain discriminator.
    :type discriminator: torch.nn.Module
    :param target_criterion: The criterion for the target model.
    :type target_criterion: callable
    :param cls_criterion: The criterion for the label classifier.
    :type cls_criterion: callable
    :param adversarial_criterion: The adversarial criterion.
    :type adversarial_criterion: callable
    :param src_batch_size: The batch size for the source domain data.
    :type src_batch_size: int
    :param device_batch_size: The batch size for the target domain data.
    :type device_batch_size: int
    :param device: the device that will be used.
    :type device: string
    :return: The average validation loss for mappings, label classification,\
             and domain classification, the average accuracy for the source\
             and the target models, and the average weighted accuracy.
    :rtype: (float, float, float, float, float, float)
    """
    mapping_loss_iterations = []
    cls_loss_iterations = []
    adversarial_loss_iterations = []

    acc_scr_epoch = []
    acc_target_epoch = []

    mini_batch_count = 1

    target_model.train(False)
    discriminator.train(False)

    with torch.no_grad():
        for x, cls_y, d_y in _get_mini_batches(
                x=val_feat, y=val_labels, src_b_size=src_batch_size,
                target_b_size=device_batch_size, device=device, shuffle_data=False):

            mapping_loss, cls_loss, adversarial_loss, acc_src, acc_trgt = _forward_pass(
                x=x, y_label=cls_y, y_domain=d_y,
                source_model=source_model, target_model=target_model,
                label_classifier=label_classifier, discriminator=discriminator,
                target_criterion=target_criterion, cls_criterion=cls_criterion,
                adversarial_criterion=adversarial_criterion, src_batch_size=src_batch_size
            )

            mapping_loss_iterations.append(mapping_loss.item())
            cls_loss_iterations.append(cls_loss.item())
            adversarial_loss_iterations.append(adversarial_loss.item())

            acc_scr_epoch.append(acc_src)
            acc_target_epoch.append(acc_trgt)

            mini_batch_count += 1

        mapping_epoch_loss = math_funcs.avg(mapping_loss_iterations)
        cls_epoch_loss = math_funcs.avg(cls_loss_iterations)
        domain_epoch_loss = math_funcs.avg(adversarial_loss_iterations)

        nb_src = (mini_batch_count - 1) * src_batch_size
        nb_trg = (mini_batch_count - 1) * 2 * device_batch_size

        source_epoch_acc = math_funcs.to_percentage(acc_scr_epoch, nb_src)
        target_epoch_acc = math_funcs.to_percentage(acc_target_epoch, nb_trg)

        w_src, w_trgt = math_funcs.weighting_factors(nb_src, nb_trg)

        w_acc = w_src * source_epoch_acc + w_trgt * target_epoch_acc

        return mapping_epoch_loss, cls_epoch_loss, domain_epoch_loss, source_epoch_acc, target_epoch_acc, w_acc


def domain_adaptation(source_model, target_model, label_classifier, discriminator,
                      target_model_opt, discriminator_opt,
                      adversarial_criterion, target_criterion, cls_criterion,
                      tr_feat, tr_labels, val_feat, val_labels, epochs,
                      update_cnt, result_dir, patience, device, factor):
    """The domain adaptation step of the AUDAC method.

    :param source_model: The non-adapted source model.
    :type source_model: torch.nnn.Module
    :param target_model: The target model, to be adapted.
    :type target_model: torch.nn.Module
    :param label_classifier: The pre-trained label classifier.
    :type label_classifier: torch.nn.Module
    :param discriminator: The domain discriminator.
    :type discriminator: torch.nn.Module
    :param target_model_opt: The optimizer of the target model.
    :type target_model_opt: torch.optim.Optimizer
    :param discriminator_opt: The optimizer for the discriminator.
    :type discriminator_opt: torch.optim.Optimizer
    :param target_criterion: The criterion for the target model.
    :type target_criterion: callable
    :param cls_criterion: The criterion for the label classifier.
    :type cls_criterion: callable
    :param adversarial_criterion: The adversarial criterion.
    :type adversarial_criterion: callable
    :param tr_feat: The training input features.
    :type tr_feat: dict[str, torch.Tensor]
    :param tr_labels: The training targeted values
    :type tr_labels: dict[str, torch.Tensor]
    :param val_feat: The validation input features.
    :type val_feat: dict[str, torch.Tensor]
    :param val_labels: The validation targeted values
    :type val_labels: dict[str, torch.Tensor]
    :param epochs: The maximum amount of epochs.
    :type epochs: int
    :param update_cnt: The update counter for updating the parameters of the\
                       discriminator.
    :type update_cnt: int
    :param result_dir: The directory to store the figures.
    :type result_dir: str
    :param patience: Amount of epochs for patience.
    :type patience: int
    :param device: the device that will be used.
    :type device: string
    :param factor: the coefficient used to be multiplied by classification loss.
    :type factor: int
    :return: The adapted model.
    :rtype: torch.nn.Module
    """
    src_batch_size_tr = 10
    device_batch_size_tr = 3
    src_batch_size_val = 34
    device_batch_size_val = 6

    best_val_score = -1
    patience_count = 0

    source_model.train(False)
    label_classifier.train(False)

    hist_mppng_loss_tr = []
    hist_cls_loss_tr = []
    hist_d_loss_tr = []
    hist_src_acc_tr = []
    hist_trgt_acc_tr = []
    hist_w_acc_tr = []

    hist_mppng_loss_val = []
    hist_cls_loss_val = []
    hist_d_loss_val = []
    hist_src_acc_val = []
    hist_trgt_acc_val = []
    hist_w_acc_val = []

    for epoch in range(epochs):
        mppng_loss_tr, cls_loss_tr, d_loss_tr, src_acc_tr, trgt_acc_tr, w_acc_tr = _training(
            tr_feat=tr_feat, tr_labels=tr_labels,
            source_model=source_model, target_model=target_model,
            label_classifier=label_classifier, discriminator=discriminator,
            target_criterion=target_criterion, cls_criterion=cls_criterion,
            adversarial_criterion=adversarial_criterion,
            target_model_opt=target_model_opt, discriminator_opt=discriminator_opt,
            src_batch_size=src_batch_size_tr, device_batch_size=device_batch_size_tr,
            update_cnt=update_cnt, device=device, factor=factor
        )

        mppng_loss_val, cls_loss_val, d_loss_val, src_acc_val, trgt_acc_val, w_acc_val = _validation(
            val_feat=val_feat, val_labels=val_labels,
            source_model=source_model, target_model=target_model,
            label_classifier=label_classifier, discriminator=discriminator,
            target_criterion=target_criterion, cls_criterion=cls_criterion,
            adversarial_criterion=adversarial_criterion,
            src_batch_size=src_batch_size_val, device_batch_size=device_batch_size_val, device=device
        )

        hist_mppng_loss_tr.append(mppng_loss_tr)
        hist_cls_loss_tr.append(cls_loss_tr)
        hist_d_loss_tr.append(d_loss_tr)
        hist_src_acc_tr.append(src_acc_tr)
        hist_trgt_acc_tr.append(trgt_acc_tr)
        hist_w_acc_tr.append(w_acc_tr)

        hist_mppng_loss_val.append(mppng_loss_val)
        hist_cls_loss_val.append(cls_loss_val)
        hist_d_loss_val.append(d_loss_val)
        hist_src_acc_val.append(src_acc_val)
        hist_trgt_acc_val.append(trgt_acc_val)
        hist_w_acc_val.append(w_acc_val)

        loss_plot = {
            'mapping': hist_mppng_loss_tr, 'class': hist_cls_loss_tr, 'adversarial': hist_d_loss_tr,
            'val_mapping': hist_mppng_loss_val, 'val_class': hist_cls_loss_val, 'val_adversarial': hist_d_loss_val
        }

        acc_plot = {
            'source': hist_src_acc_tr, 'target': hist_trgt_acc_tr, 'weighted': hist_w_acc_tr,
            'val_source': hist_src_acc_val, 'val_target': hist_trgt_acc_val, 'val_weighted': hist_w_acc_val
        }

        plot_function(loss_plot, acc_plot, result_dir)

        printing.domain_adaptation_msg(
            epoch=epoch + 1, epochs=epochs,
            mppng_loss_tr=mppng_loss_tr, cls_loss_tr=cls_loss_tr, domain_loss_tr=d_loss_tr,
            mppng_loss_val=mppng_loss_val, cls_loss_val=cls_loss_val, domain_loss_val=d_loss_val,
            src_acc_tr=src_acc_tr, trgt_acc_tr=trgt_acc_tr, w_acc_tr=w_acc_tr,
            src_acc_val=src_acc_val, trgt_acc_val=trgt_acc_val, w_acc_va=w_acc_val
        )

        if best_val_score < trgt_acc_val:
            best_val_score = trgt_acc_val
            file_io.save_pytorch_model(target_model, 'target_cnn', result_dir)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count > patience:
                break

    printing.training_stopping_msg(best_val_score)

# EOF
