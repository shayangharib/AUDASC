#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

__author__ = 'Shayan Gharib, Konstantinos Drossos -- TUT'
__docformat__ = 'reStructuredText'
__all__ = [
    'domain_adaptation_msg',
    'info_msg',
    'pre_training_msg'
]


def get_data_msg(tr, val, test, title, ending='\n', flushing=True):

    """Prints the number of samples and shape of training, validation, and test set.

    :param tr: training features.
    :type val: validation features.
    :param test: test features.
    :type title: a string as title to show the device that the data are coming from e.g. A, B, or C
    :param ending: The ending of the complete message (including\
                   decorations).
    :type ending: str
    :param flushing: Flush the buffer now?
    :type flushing: bool
    """
    rows = list()
    rows.append('\n{}'.format(title))
    rows.append('\n{}\n'.format('*' * len(title)))
    rows.append('train:{}'.format(tr.shape))
    rows.append('validation:{}'.format(val.shape))
    rows.append('test:{}'.format(test.shape))
    rows.append('\n{}\n'.format('*' * len(title)))

    print(''.join(rows), end=ending, flush=flushing)


def domain_adaptation_msg(epoch, epochs,
                          mppng_loss_tr, cls_loss_tr, domain_loss_tr,
                          mppng_loss_val, cls_loss_val, domain_loss_val,
                          src_acc_tr, trgt_acc_tr, w_acc_tr,
                          src_acc_val, trgt_acc_val, w_acc_va, flush=True):
    """Prints the results of the domain adaptation step.

    :param epoch: The epoch to print the results for.
    :type epoch: int
    :param epochs: The total amount of epochs.
    :type epochs: int
    :param mppng_loss_tr: The mapping training loss.
    :type mppng_loss_tr: float
    :param cls_loss_tr: The classification training loss.
    :type cls_loss_tr: float
    :param domain_loss_tr: The domain classification training loss.
    :type domain_loss_tr: float
    :param mppng_loss_val: The mapping validation loss.
    :type mppng_loss_val: float
    :param cls_loss_val: The classification validation loss.
    :type cls_loss_val: float
    :param domain_loss_val: The domain classification validation loss.
    :type domain_loss_val: float
    :param src_acc_tr: The training accuracy of the source domain data.
    :type src_acc_tr: float
    :param trgt_acc_tr: The training accuracy of the target domain data.
    :type trgt_acc_tr: float
    :param w_acc_tr: The weighted training accuracy.
    :type w_acc_tr: float
    :param src_acc_val: The validation accuracy of the source domain data.
    :type src_acc_val: float
    :param trgt_acc_val: The validation accuracy of the target domain data.
    :type trgt_acc_val: float
    :param w_acc_va: The weighted validation accuracy.
    :type w_acc_va: float
    :param flush: Flush buffer now?
    :type flush: bool
    """
    print('Epoch: {e:3d}/{e_s:3d} | '
          'Losses (Tr/Va) -- Map: {m_l_tr:4.2f}/{m_l_va:4.2f}, '
          'Cls: {cls_l_tr:5.3f}/{cls_l_va:5.3f}, Adv: {adv_l_tr:4.2f}/{adv_l_va:4.2f} | '
          'Accuracy (Tr/Va) -- Src: {acc_src_tr:6.2f}/{acc_src_va:5.2f}, '
          'Trgt: {acc_trgt_tr:5.2f}/{acc_trgt_va:5.2f}, Weighted: {acc_w_tr:5.2f}/{acc_w_va:5.2f}'.format(
            e=epoch, e_s=epochs,
            m_l_tr=mppng_loss_tr,
            m_l_va=mppng_loss_val,
            cls_l_tr=cls_loss_tr,
            cls_l_va=cls_loss_val,
            adv_l_tr=domain_loss_tr,
            adv_l_va=domain_loss_val,
            acc_src_tr=src_acc_tr,
            acc_src_va=src_acc_val,
            acc_trgt_tr=trgt_acc_tr,
            acc_trgt_va=trgt_acc_val,
            acc_w_tr=w_acc_tr,
            acc_w_va=w_acc_va),
          flush=flush)


def info_msg(the_msg, ending='\n', flushing=True):
    """Prints the message in an informative way that we like.

    :param the_msg: The message to be printed.
    :type the_msg: str
    :param ending: The ending of the complete message (including\
                   decorations).
    :type ending: str
    :param flushing: Flush the buffer now?
    :type flushing: bool
    """
    rows = list()

    rows.append('\n**{}**\n'.format('*' * len(the_msg)))
    rows.append('* {} *\n'.format(the_msg))
    rows.append('**{}**\n'.format('*' * len(the_msg)))

    print(''.join(rows), end=ending, flush=flushing)


def pre_training_msg(epoch, epochs, epoch_tr_loss, epoch_val_loss, tr_acc, val_acc):
    """Prints information about the epoch.

    :param epoch: The epoch to print information about.
    :type epoch: int
    :param epochs: The total amount of epochs.
    :type epochs: int
    :param epoch_tr_loss: The training loss for the epoch.
    :type epoch_tr_loss: float
    :param epoch_val_loss: The valiation loss for the epoch.
    :type epoch_val_loss: float
    :param tr_acc: The training accuracy for the epoch.
    :type tr_acc: float
    :param val_acc: The validation accuracy for the epoch.
    :type val_acc: float
    """
    print(
        'Epoch: {:3d}/{:3d}, '
        'loss: {:7.3f}, val_loss: {:7.3f}, '
        'acc: {:7.2f}, val_acc: {:7.2f}'.format(
            epoch, epochs,
            epoch_tr_loss, epoch_val_loss,
            tr_acc, val_acc
        ),
        flush=True
    )


def training_stopping_msg(best_val):
    """Prints a message that we stopping the training.

    :param best_val: The value achieved.
    :type best_val: float
    """
    print(
        '\nStopping training, validation accuracy not '
        'improving after {:.2f}\n'.format(best_val),
        flush=True
    )


def testing_result_msg(non_adapted_res, adapted_res, ending='\n', flushing=True):
    """Prints the result of test phase.

    :param non_adapted_res: The values achieved for non adapted model.
    :type non_adapted_res: Dict
    :param adapted_res: The values achieved for adapted model.
    :type adapted_res: Dict
    :param ending: The ending of the complete message (including\
                   decorations).
    :type ending: str
    :param flushing: Flush the buffer now?
    :type flushing: bool
    """
    rows = list()
    rows.append('\nThe performance of non-adapted model:')
    rows.append('\nSource --- device A: {:.2f}'.format(non_adapted_res['A']))
    rows.append('\nTarget --- devices B & C: {:.2f}'.format(non_adapted_res['BC']))
    rows.append('\nAll devices: {:.2f}'.format(non_adapted_res['all']))
    rows.append('\n')
    rows.append('\n')
    rows.append('\nThe performance of adapted model:')
    rows.append('\nSource --- device A: {:.2f}'.format(adapted_res['A']))
    rows.append('\nTarget --- devices B & C: {:.2f}'.format(adapted_res['BC']))
    rows.append('\nAll devices: {:.2f}'.format(adapted_res['all']))

    print(''.join(rows), end=ending, flush=flushing)

# EOF
