from __future__ import annotations

from typing import Union, List

from torch import Tensor

from data.dataset import Dataset
import torch
import numpy as np
from ml.method import Method
from logging import getLogger
from ml.common import TrainEpoch, TrainPerformance, EarlyStopping, General
from ml.summary_writer import MySummaryWriter
import random

logger = getLogger('app')


class MLTrainer(object):

    def __init__(self, dataset: Dataset, method: Method, train_params: dict):
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        self.dataset = dataset
        self.method = method
        self._train_params = train_params

    def run(self, train_ids: list, validation_ids: list,
            writer: MySummaryWriter = None) -> TrainPerformance:
        """
        Train & validate predictor for one set of parameters and ids
        :param writer:
        :param train_ids:
        :param validation_ids:
        :param batch_size:
        :param epochs:
        :param is_early_stopping:
        :return:
        """
        method = self.method
        train_params = self._train_params

        epochs = train_params['epochs']
        batch_size = train_params['batch_size']
        early_stopping = None
        if train_params['early_stopping']:
            assert 'early_stopping_patience' in train_params, 'early_stopping_patience needs to be configured' \
                                                              ' to use early stopping'
            early_stopping = EarlyStopping(patience=train_params['early_stopping_patience'], delta=0.01)

        train_set = method.get_dataset(ids=train_ids, dataset=self.dataset)
        validation_set = method.get_dataset(ids=validation_ids, dataset=self.dataset)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                                                   worker_init_fn=MyWorkerInit())
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True,
                                                        pin_memory=True, worker_init_fn=MyWorkerInit())
        performance = TrainPerformance()
        num_epochs = 0
        for epoch_id in range(epochs):
            logger.debug("Epoch {}".format(epoch_id))

            train_epoch = self._train_epoch(method=method, loader=train_loader,
                                            epoch_id=epoch_id, writer=writer)
            val_epoch = self._validate_epoch(method=method, loader=validation_loader,
                                             epoch_id=epoch_id, writer=writer)

            logger.debug("Training performance " + str(train_epoch))
            logger.debug("Validation performance " + str(val_epoch))

            # append average performance for this epoch
            performance.add_epoch_performance(train_epoch=train_epoch, validate_epoch=val_epoch)
            num_epochs += 1

            if writer is not None:
                writer.add_performance_epoch(train_epoch=train_epoch,
                                             val_epoch=val_epoch, epoch_id=epoch_id)

            # stop training if F1 score doesn't improve anymore
            if early_stopping is not None:
                eval_val = val_epoch.metrics["f1"] * (-1)
                # eval_val = val_loss
                early_stopping(eval_val, method.model)
                if early_stopping.early_stop:
                    break

        if early_stopping is not None:
            method.model = early_stopping.best_model

        logger.info("Training performance " + str(performance.train_metrics[-1]))
        logger.info("Validation performance " + str(performance.validate_metrics[-1]))

        return performance

    def _train_epoch(self, loader: torch.utils.data.DataLoader,
                     method: Method, epoch_id: int,
                     writer: MySummaryWriter = None) -> TrainEpoch:
        epoch = TrainEpoch(_id=epoch_id)
        method.model.train()
        # feature_batch.shape=(B, 2*T + 1025, T)
        # target_batch.shape=(B, 3, T)
        # loss_mask_batch.shape=(B, 3, T)
        for feature_batch, padding_batch, target_batch, loss_mask_batch, _ in loader:
            method.optimizer.zero_grad()
            loss_norm = self._forward(feature_batch=feature_batch, padding_batch=padding_batch,
                                      target_batch=target_batch,
                                      loss_mask_batch=loss_mask_batch,
                                      method=method, epoch=epoch, writer=writer)
            loss_norm.backward()
            method.optimizer.step()

        epoch.normalize()
        return epoch

    def _validate_epoch(self, loader: torch.utils.data.DataLoader,
                        method: Method, epoch_id: int,
                        writer: MySummaryWriter = None) -> TrainEpoch:
        epoch = TrainEpoch(_id=epoch_id)

        method.model.eval()
        with torch.no_grad():
            for feature_batch, padding_batch, target_batch, loss_mask_batch, _ in loader:
                loss_norm = self._forward(feature_batch=feature_batch, target_batch=target_batch,
                                          loss_mask_batch=loss_mask_batch, padding_batch=padding_batch,
                                          method=method, epoch=epoch, writer=writer)

        epoch.normalize()
        return epoch

    def _forward(self, feature_batch: Union[Tensor, List[Tensor]], padding_batch: Tensor,
                 target_batch: Tensor, loss_mask_batch: Tensor, method: Method,
                 epoch: TrainEpoch, writer: MySummaryWriter = None) -> torch.Tensor:
        pred_batch = method.forward(feature_batch=feature_batch)
        target_batch = target_batch.to(self.device)
        loss_mask_batch = loss_mask_batch.to(self.device)
        # don't consider padded positions for loss calculation
        # normalize based on length
        loss_el_batch = method.loss(pred_batch, target_batch)  # loss_el_batch.shape=(B, 3, T)
        loss_el_masked_batch = loss_el_batch * loss_mask_batch  # loss_el_masked_batch.shape=(B, 3, T)
        loss_norm = torch.sum(loss_el_masked_batch)  # loss_norm.shape=(1)

        # add to performance object
        epoch.metrics["loss"] += loss_norm.item()
        epoch.metrics["loss_count"] += loss_el_masked_batch.shape[0]
        epoch.add_batch(padding_batch=padding_batch,
                        pred_batch=pred_batch,
                        target_batch=target_batch)
        if writer is not None:
            writer.add_model(model=method.model, feature_batch=feature_batch)

        return loss_norm


class MyWorkerInit(object):
    def __call__(self, worker_id):
        # https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
        # There is no need at the moment. Just in case we use random in the TorchDataset.
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
