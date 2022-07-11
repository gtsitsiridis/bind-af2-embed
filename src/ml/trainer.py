from __future__ import annotations

from torch import Tensor

from data.dataset import Dataset
import torch
import numpy as np
from ml.method import Method
from logging import getLogger
from ml.common import PerformanceMap, EarlyStopping, General, Results, ProteinResult
from ml.summary_writer import MySummaryWriter
import random
from pathlib import Path
from enum import Enum
import tracemalloc
from hurry.filesize import size as psize

from ml.template import RunTemplate

logger = getLogger('app')


class TrainMode(Enum):
    BINDING = "BINDING"
    LIGAND = "LIGAND"


class MLTrainer(object):

    def __init__(self, dataset: Dataset, method: Method, template: RunTemplate,
                 train_writer: MySummaryWriter = None,
                 val_writer: MySummaryWriter = None,
                 performance_file_path: Path = None,
                 results_file_path: Path = None,
                 model_file_path: Path = None,
                 log_tracemalloc: bool = False):
        if torch.cuda.is_available():
            self._device = 'cuda:0'
        else:
            self._device = 'cpu'

        self._dataset = dataset
        self._method = method
        self._train_params = template.train_params
        self._train_writer = train_writer
        self._val_writer = val_writer
        self._performance_file_path = performance_file_path
        self._model_file_path = model_file_path
        self._results_file_path = results_file_path
        self._early_stopping_metric = 'f1_total'
        self._mode = TrainMode[template.train_params['train_mode']]
        if self._mode == TrainMode.BINDING:
            self._early_stopping_metric = 'f1_binding'
        self._log_tracemalloc = log_tracemalloc
        if log_tracemalloc:
            tracemalloc.start()

    def __call__(self, train_ids: list, validation_ids: list) -> (Results, Results):
        """
        Train & validate predictor for one set of parameters and ids
        :param train_ids:
        :param validation_ids:
        :return: train and validation results of last epoch
        """
        method = self._method
        train_writer = self._train_writer
        val_writer = self._val_writer
        train_params = self._train_params

        epochs = train_params['epochs']
        batch_size = train_params['batch_size']
        early_stopping = None
        if train_params['early_stopping']:
            early_stopping = EarlyStopping(patience=train_params.get('early_stopping_patience', 10),
                                           delta=0.01)

        train_set = method.get_dataset(ids=train_ids, writer=train_writer)
        validation_set = method.get_dataset(ids=validation_ids, writer=val_writer)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                                                   worker_init_fn=MyWorkerInit())
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True,
                                                        pin_memory=True, worker_init_fn=MyWorkerInit())
        performance_map = PerformanceMap()
        train_results = None
        validation_results = None
        for epoch_id in range(epochs):
            self._log_memory('started epoch')
            logger.info("Epoch {}".format(epoch_id))
            train_results = self._train_epoch(loader=train_loader,
                                              epoch_id=epoch_id)
            validation_results = self._validate_epoch(loader=validation_loader,
                                                      epoch_id=epoch_id)
            self._log_memory('logging epoch performance')
            # get performance
            epoch_train_performance = train_results.get_performance(cutoff=train_params['cutoff'], is_train=True)
            epoch_validation_performance = validation_results.get_performance(cutoff=train_params['cutoff'],
                                                                              is_train=True)

            performance_map.append_performance(performance=epoch_train_performance,
                                               tag=f'train_epoch_{str(epoch_id)}')
            performance_map.append_performance(performance=epoch_validation_performance,
                                               tag=f'val_epoch_{str(epoch_id)}')

            # log performance
            tag = 'total'
            if self._mode == TrainMode.BINDING:
                tag = 'binding'
            logger.info("Training performance: " + epoch_train_performance.to_str(tag))
            logger.info("Validation performance: " + epoch_validation_performance.to_str(tag))
            if train_writer is not None:
                train_writer.add_performance_scalars(performance=epoch_train_performance,
                                                     epoch_id=epoch_id)
                if epoch_id % 10 == 0:
                    train_writer.add_protein_results(train_results, cutoff=train_params['cutoff'], epoch=epoch_id)
            if val_writer is not None:
                val_writer.add_performance_scalars(performance=epoch_validation_performance,
                                                   epoch_id=epoch_id)
                if epoch_id % 10 == 0:
                    val_writer.add_protein_results(validation_results, cutoff=train_params['cutoff'], epoch=epoch_id)
            self._log_memory('logged epoch performance')
            # stop training if F1 score doesn't improve anymore
            if early_stopping is not None:
                eval_val = epoch_validation_performance[self._early_stopping_metric] * (-1)
                # eval_val = val_loss
                early_stopping(eval_val, method.model)
                if early_stopping.early_stop:
                    break

        if early_stopping is not None:
            method.model = early_stopping.best_model

        if self._model_file_path is not None:
            torch.save(method.model, self._model_file_path)

        if self._performance_file_path is not None:
            General.to_csv(df=performance_map.to_df(),
                           filename=self._performance_file_path)
        if self._results_file_path is not None:
            General.to_csv(df=validation_results.to_df(cutoff=train_params['cutoff']),
                           filename=self._results_file_path)

        if self._log_tracemalloc:
            tracemalloc.stop()

        return train_results, validation_results

    def _train_epoch(self, loader: torch.utils.data.DataLoader,
                     epoch_id: int) -> Results:
        method = self._method
        results = Results()

        method.model.train()
        # feature_batch.shape=(B, 2*T + 1025, T)
        # target_batch.shape=(B, 3, T)
        # loss_mask_batch.shape=(B, 3, T)
        self._log_memory('loading train batch')
        for feature_batch, padding_batch, target_batch, loss_mask_batch, prot_ids in loader:
            self._log_memory('loaded train batch')
            method.optimizer.zero_grad()
            target_batch = target_batch.to(self._device)
            loss_mask_batch = loss_mask_batch.to(self._device)
            self._log_memory('forward pass')
            pred_batch = method.forward(feature_batch=feature_batch)
            self._log_memory('done backward pass')
            loss_batch = method.loss(pred_batch=pred_batch, loss_mask_batch=loss_mask_batch, target_batch=target_batch)
            loss_norm = torch.sum(loss_batch)
            self._log_memory('backward pass')
            loss_norm.backward()
            self._log_memory('done backward pass')
            method.optimizer.step()
            self._log_memory('done optmizer')

            batch_results = self._batch_results(padding_batch=padding_batch, prot_ids=prot_ids, pred_batch=pred_batch,
                                                target_batch=target_batch, loss_batch=loss_batch,
                                                tag=f'train_epoch_{epoch_id}')
            results.append(batch_results)
        return results

    def _validate_epoch(self, loader: torch.utils.data.DataLoader,
                        epoch_id: int) -> Results:
        method = self._method
        method.model.eval()
        results = Results()
        self._log_memory('loading val batch')
        with torch.no_grad():
            for feature_batch, padding_batch, target_batch, loss_mask_batch, prot_ids in loader:
                self._log_memory('loaded val batch')
                target_batch = target_batch.to(self._device)
                loss_mask_batch = loss_mask_batch.to(self._device)
                pred_batch = method.forward(feature_batch=feature_batch)
                loss_batch = method.loss(pred_batch=pred_batch, loss_mask_batch=loss_mask_batch,
                                         target_batch=target_batch)

                batch_results = self._batch_results(padding_batch=padding_batch, prot_ids=prot_ids,
                                                    pred_batch=pred_batch,
                                                    target_batch=target_batch, loss_batch=loss_batch,
                                                    tag=f'validation_epoch_{epoch_id}')
                results.append(batch_results)

        return results

    @staticmethod
    def _batch_results(padding_batch: Tensor, target_batch: Tensor, pred_batch: Tensor,
                       loss_batch: Tensor, prot_ids: tuple, tag: str) -> Results:
        batch_results = Results()
        for idx in range(len(padding_batch)):
            prot_id = prot_ids[idx]
            target = target_batch[idx]
            padding = padding_batch[idx]
            predictions = pred_batch[idx]
            loss = loss_batch[idx]
            result = ProteinResult(prot_id=prot_id, padding=padding, target=target, predictions=predictions,
                                   loss=loss, tag=tag)
            batch_results[prot_id] = result

        return batch_results

    def _log_memory(self, state: str):
        if self._log_tracemalloc:
            size, peak = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            logger.debug("Memory state: " + state)
            logger.debug(f"{psize(size)=}, {psize(peak)=}")


class MyWorkerInit(object):
    def __call__(self, worker_id):
        # https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
        # There is no need at the moment. Just in case we use random in the TorchDataset.
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
