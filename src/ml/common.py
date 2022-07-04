from __future__ import annotations

import torch
import numpy as np
import math
from typing import List, Union
import pandas as pd
from typing import Dict
from logging import getLogger
from plots import Plots

from torch import Tensor
from data.dataset import BindAnnotation
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from plots import Plots
from pathlib import Path

logger = getLogger('app')


class General(object):
    @staticmethod
    def to_csv(df: pd.DataFrame, filename: Path, append: bool = False):
        if append:
            df.to_csv(str(filename), index=False, mode='a', header=False)
        else:
            df.to_csv(str(filename), index=False)

    @staticmethod
    def remove_padded_positions(pred, target, padding, loss):
        indices = (padding != 0).nonzero()

        pred_i = pred[:, indices].squeeze()
        target_i = target[:, indices].squeeze()
        loss_i = None
        if loss is not None:
            loss_i = loss[:, indices].squeeze()

        return pred_i, target_i, loss_i

    @staticmethod
    def forward(feature_batch: Union[Tensor, List[Tensor]], device: str, model: torch.nn.Module) -> Tensor:
        # feature_batch.shape=(B, 2*T + 1024, T)
        if type(feature_batch) is list:
            for i, feature_batch_i in enumerate(feature_batch):
                feature_batch[i] = feature_batch_i.to(device)
            pred_batch = model.forward(*feature_batch)  # pred_batch.shape=(B, 3, T)
        else:
            feature_batch = feature_batch.to(device)
            pred_batch = model.forward(feature_batch)

        return pred_batch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, delta=0, ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_model = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
            self.best_model = model
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.debug(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.best_model = model
            self.counter = 0


class ProteinResult(object):

    def __init__(self, prot_id: str, padding: torch.Tensor, predictions: torch.Tensor, target: torch.Tensor,
                 loss: torch.Tensor, tag: str):
        assert target.shape == predictions.shape, "The target and prediction arrays need to have the same shape"
        predictions = torch.sigmoid(predictions)
        predictions, target, loss = General.remove_padded_positions(predictions.detach().numpy(),
                                                                    target.detach().numpy(),
                                                                    padding.detach().numpy(),
                                                                    loss.detach().numpy())

        self._prot_id = prot_id
        self._target = target
        self._predictions = predictions
        self._tag = tag
        self._loss = loss

    @property
    def tag(self):
        return self._tag

    @property
    def prot_id(self):
        return self._prot_id

    def add(self, prot_result: ProteinResult):
        self._predictions = np.add(self._predictions, np.around(prot_result._predictions, 3))
        self._loss = np.add(self._loss, np.around(prot_result._loss, 3))

    def normalize(self, norm_factor):
        self._predictions = np.around(self._predictions / norm_factor, 3)
        self._loss = np.around(self._loss / norm_factor, 3)

    def plot_confusion_matrix(self, _class: int, cutoff: float) -> plt.figure:
        pred_copy = self.prediction_to_labels(cutoff=cutoff)
        target = self._target

        label = BindAnnotation.id2name(_class)

        # Build confusion matrix
        cf_matrix = confusion_matrix(target[_class], pred_copy[_class])
        if cf_matrix.shape == (1, 1):
            return None

        return Plots.plot_confusion_matrix(cf_matrix, label, ["N", "P"])

    def prediction_to_ri(self, cutoff: float) -> np.array:
        return np.abs(self._predictions - cutoff) / cutoff * 9

    def prediction_to_labels(self, cutoff: float) -> np.array:
        pred_copy = np.zeros(self._predictions.shape, dtype=np.int32)
        pred_copy[self._predictions >= cutoff] = 1
        return pred_copy

    def to_df(self, cutoff: float) -> pd.DataFrame:
        prot_ids = []
        positions = []
        targets = []
        probabilities = []
        ligands = []
        ris = []
        cutoffs = []
        predictions = []
        tags = []
        losses = []

        tags.extend([self._tag] * len(self) * 4)
        cutoffs.extend([cutoff] * len(self) * 4)
        target = self._target.flatten()
        targets.extend(target.tolist())
        positions.extend(list(range(len(self))) * 4)
        prot_ids.extend([self._prot_id] * len(self) * 4)
        ligands.extend(([BindAnnotation.id2name(0)] * len(self)) +
                       ([BindAnnotation.id2name(1)] * len(self)) +
                       ([BindAnnotation.id2name(2)] * len(self)) +
                       ([BindAnnotation.id2name(3)] * len(self)))

        prediction = self._predictions.flatten()
        probabilities.extend(prediction.tolist())
        ris.extend(self.prediction_to_ri(cutoff=cutoff).flatten())
        predictions.extend(self.prediction_to_labels(cutoff=cutoff).flatten())
        losses.extend(self._loss.flatten())

        return pd.DataFrame(
            zip(tags, prot_ids, positions, ligands, targets, predictions, cutoffs, ris, probabilities, losses),
            columns=['tag', 'protd_id', 'position', 'ligand', 'target', 'prediction', 'cutoff', 'ri',
                     'prob', 'loss'])

    def __len__(self):
        return self._target.shape[1]


class Results(object):
    def __init__(self):
        self.results_dict: Dict[str, ProteinResult] = {}

    def merge(self, results_dict: Results):
        self.results_dict = {**self.results_dict, **results_dict}

    def keys(self):
        return self.results_dict.keys()

    def append(self, results: Results):
        self.results_dict.update(results)

    def __getitem__(self, item) -> ProteinResult:
        return self.results_dict[item]

    def __setitem__(self, key: str, value: ProteinResult):
        self.results_dict[key] = value

    def __len__(self):
        len(self.results_dict)

    def to_df(self, cutoff: float) -> pd.DataFrame:
        return pd.concat(list(map(lambda prot_result: prot_result.to_df(cutoff=cutoff),
                                  list(self.results_dict.values()))),
                         axis=0)

    def get_single_performance(self, cutoff: float, tag_value: str = None) -> SinglePerformance:
        metrics = {}
        df = self.to_df(cutoff=cutoff)

        if tag_value is not None:
            df = df[df['tag'] == tag_value]

        # ligand metrics
        ligands = df.ligand.unique()
        for ligand in ligands:
            df_ligand = df[df.ligand == ligand]
            metrics['loss' + "_" + ligand] = df_ligand.loss.mean()
            metrics['acc' + "_" + ligand], metrics['prec' + "_" + ligand], metrics['rec' + "_" + ligand], metrics[
                'f1' + "_" + ligand], metrics['mcc' + "_" + ligand] = \
                self.calc_performance_measurements(df=df_ligand)

        # total metrics
        metrics['loss_total'] = df.loss.mean()
        metrics['acc_total'], metrics['prec_total'], metrics['rec_total'], metrics['f1_total'], metrics['mcc_total'] = \
            self.calc_performance_measurements(df=df)

        return SinglePerformance(metrics=metrics)

    @staticmethod
    def calc_performance_measurements(df: pd.DataFrame):
        """Calculate precision, recall, f1, mcc, and accuracy"""

        tp = fp = tn = fn = 0
        counts = df[['target', 'prediction']].value_counts()
        if (1, 1) in counts.index:
            tp = counts.loc[(1, 1)] / sum(counts)
        if (0, 1) in counts.index:
            fp = counts.loc[(0, 1)] / sum(counts)
        if (0, 0) in counts.index:
            tn = counts.loc[(0, 0)] / sum(counts)
        if (1, 0) in counts.index:
            fn = counts.loc[(1, 0)] / sum(counts)

        tp = float(tp)
        fp = float(fp)
        fn = float(fn)
        tn = float(tn)

        recall = prec = f1 = mcc = 0
        acc = round((tp + tn) / (tp + tn + fn + fp), 3)

        if tp > 0 or fn > 0:
            recall = round(tp / (tp + fn), 3)
        if tp > 0 or fp > 0:
            prec = round(tp / (tp + fp), 3)
        if recall > 0 or prec > 0:
            f1 = round(2 * recall * prec / (recall + prec), 3)
        if (tp > 0 or fp > 0) and (tp > 0 or fn > 0) and (tn > 0 or fp > 0) and (tn > 0 or fn > 0):
            mcc = round((tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)), 3)

        return acc, prec, recall, f1, mcc


class SinglePerformance(object):
    def __init__(self, metrics: dict):
        self._metrics = metrics

    def keys(self):
        return self._metrics.keys()

    def __getitem__(self, item):
        return self._metrics[item]

    def __str__(self):
        return "Loss: {:.3f}, Prec: {:.3f}, Recall: {:.3f}, F1: {:.3f}, MCC: {:.3f}".format(self["loss_total"],
                                                                                            self["prec_total"],
                                                                                            self["rec_total"],
                                                                                            self["f1_total"],
                                                                                            self["mcc_total"])


class Performance(object):
    def __init__(self, cutoff: float):
        self._cutoff = cutoff
        self._metrics = {'tag': []}

    def get_single_performance(self, idx: int) -> SinglePerformance:
        assert 0 <= idx < len(self), 'idx is out of range'
        return SinglePerformance({k: v[idx] for k, v in self._metrics.items()})

    def append_single_performance(self, single_performance: SinglePerformance, tag: str):
        metrics = self._metrics
        metrics['tag'].append(tag)
        for k in single_performance.keys():
            if k not in self._metrics:
                metrics[k] = [single_performance[k]]
            else:
                metrics[k].append(single_performance[k])

    def keys(self):
        return self._metrics.keys()

    def to_df(self):
        return pd.DataFrame(self._metrics)

    def __getitem__(self, item):
        return self._metrics[item]

    def __len__(self):
        return len(self._metrics['loss'])
