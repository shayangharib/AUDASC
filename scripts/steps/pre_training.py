#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import shuffle
import torch

from aux import file_io, printing

__author__ = 'Shayan Gharib, Konstantinos Drossos -- TUT'
__docformat__ = 'reStructuredText'
__all__ = ['pre_training']


def _training(model, label_classifier, optimizer, tr_feat, tr_labels, batch_size, criterion):
    """Implemented the training stage of the pre-training step.

    :param model: The source model, M.
    :type model: torch.nn.Module
    :param label_classifier: The label classifier, C.
    :type label_classifier: torch.nn.Module
    :param optimizer: The optimizer.
    :type optimizer: torch.optim.Optimizer
    :param tr_feat: The input data.
    :type tr_feat: torch.Tensor
    :param tr_labels: The ground truths/targeted output data.
    :type tr_labels: torch.Tensor
    :param batch_size: The batch size to be used.
    :type batch_size: int
    :param criterion: The criterion/loss function.
    :type criterion: callable
    :return: The loss and accuracy for the training data.
    :rtype: float, float
    """

    assert len(tr_feat) == len(tr_labels)

    tr_indices = list(range(len(tr_feat)))
    shuffle(tr_indices)

    x_tr = tr_feat[tr_indices]
    y_tr = tr_labels[tr_indices]

    # making mini batches
    x_tr = [x_tr[i:i+batch_size] for i in range(0, len(x_tr), batch_size)]
    y_tr = [y_tr[i:i+batch_size] for i in range(0, len(y_tr), batch_size)]

    # 5510 / 38 (batch_size) = 145 mini-batches
    # assert len(x_tr) == 145, 'pre-training -> mini_batches'

    model.train(True)
    label_classifier.train(True)

    tr_correct = []
    tr_len = []
    iter_tr_losses = []

    for x_in, y_in in zip(x_tr, y_tr):
        x_in = x_in.float()
        y_in = y_in.long()

        optimizer.zero_grad()

        source_mappings = model(x_in)
        y_hat = label_classifier(source_mappings)

        y_in = torch.argmax(y_in, dim=1)

        loss = criterion(y_hat, y_in)
        loss.backward()
        optimizer.step()

        y_hat = torch.argmax(y_hat, dim=1)

        iter_tr_losses.append(loss.item())
        tr_correct.append((y_hat == y_in).float().sum())
        tr_len.append(len(y_hat))

    tr_loss = sum(iter_tr_losses) / len(iter_tr_losses)
    tr_acc = 100 * float(sum(tr_correct)) / sum(tr_len)

    return tr_loss, tr_acc


def _validation(model, label_classifier, val_feat, val_labels, batch_size, criterion):
    """Implements the validation stage of the pre-training step.

    :param model: The source model, M.
    :type model: torch.nn.Module
    :param label_classifier: The label classifier, C.
    :type label_classifier: torch.nn.Module
    :param val_feat: The input data.
    :type val_feat: list[torch.Tensor]
    :param val_labels: The ground truths/targeted output data.
    :type val_labels: list[torch.Tensor]
    :param criterion: The criterion/loss function.
    :type criterion: callable
    :return: The loss and accuracy for the validation data.
    :rtype: float, float
    """
    # making mini batches
    x_val = [val_feat[i:i+batch_size] for i in range(0, len(val_feat), batch_size)]
    y_val = [val_labels[i:i+batch_size] for i in range(0, len(val_labels), batch_size)]

    # 612 / 36(batch_size) = 17 mini-batches
    # assert len(x_val) == 17, 'source model -> validation part -> mini_batches'

    val_loss = []
    val_correct = []
    val_len = []

    model.train(False)
    label_classifier.train(False)

    for x_in, y_in in zip(x_val, y_val):
        x_in = x_in.float()
        y_in = y_in.long()

        y_hat = label_classifier(model(x_in))

        y_in = torch.argmax(y_in, dim=1)

        val_loss.append(criterion(y_hat, y_in).item())

        y_hat = torch.argmax(y_hat, dim=1)

        val_correct.append((y_hat == y_in).float().sum())
        val_len.append(len(y_hat))

    epoch_val_loss = sum(val_loss) / len(val_loss)
    val_acc = 100 * float(sum(val_correct)) / sum(val_len)

    return epoch_val_loss, val_acc


def pre_training(model, label_classifier, optimizer,
                 tr_batch_size, val_batch_size, tr_feat, tr_labels, val_feat, val_labels,
                 epochs, criterion, patience, result_dir):
    """The pre-training step of the method.

    :param model: The model, M.
    :type model: torch.nn.Module
    :param label_classifier: The label classifier, C.
    :type label_classifier: torch.nn.Module
    :param optimizer: The optimizer.
    :type optimizer: torch.optim.Optimizer
    :param tr_batch_size: The batch size to be used at the training stage.
    :type tr_batch_size: int
    :param val_batch_size: The batch size to be used at the validation stage.
    :type val_batch_size: int
    :param tr_feat: The features for training.
    :type tr_feat: torch.Tensor
    :param tr_labels: The targeted labels for training.
    :type tr_labels: torch.Tensor
    :param val_feat: The features for validation.
    :type val_feat: torch.Tensor
    :param val_labels: The targeted labels for validation.
    :type val_labels: torch.Tensor
    :param epochs: The total amount of epochs.
    :type epochs: int
    :param criterion: The criterion for the label classifier
    :type criterion: callable
    :param patience: The amount of epochs for patience.
    :type patience: int
    :param result_dir: The directory to output the saved model, label classifier,\
                       and optimizer.
    :type result_dir: dir
    """

    patience_count = 0
    best_val_acc = -1

    printing.info_msg('Pre-training phase')

    for epoch in range(epochs):

        epoch_tr_loss, epoch_tr_acc = _training(
            model=model, label_classifier=label_classifier, optimizer=optimizer,
            tr_feat=tr_feat, tr_labels=tr_labels, batch_size=tr_batch_size, criterion=criterion
        )

        epoch_val_loss, epoch_val_acc = _validation(
            model=model, label_classifier=label_classifier,
            val_feat=val_feat, val_labels=val_labels, batch_size=val_batch_size, criterion=criterion
        )

        printing.pre_training_msg(
            epoch=epoch + 1, epochs=epochs,
            epoch_tr_loss=epoch_tr_loss, epoch_val_loss=epoch_val_loss,
            tr_acc=epoch_tr_acc, val_acc=epoch_val_acc
        )

        if best_val_acc < epoch_val_acc:
            best_val_acc = epoch_val_acc
            file_io.save_pytorch_model(model, 'non_adapted_cnn', result_dir)
            file_io.save_pytorch_model(label_classifier, 'label_classifier', result_dir)
            file_io.save_pytorch_model(optimizer, 'pre_training_optimizer', result_dir)
            patience_count = 0

        else:
            patience_count += 1
            if patience_count > patience:
                break

    printing.training_stopping_msg(best_val_acc)

# EOF
