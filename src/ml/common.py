from __future__ import annotations

import torch
import numpy as np
import math
from typing import List, Union
import pandas as pd
from typing import Dict
from logging import getLogger

from torch import Tensor

logger = getLogger('app')


class General(object):
    @staticmethod
    def remove_padded_positions(pred, target, padding):
        indices = (padding != 0).nonzero()

        pred_i = pred[:, indices].squeeze()
        target_i = target[:, indices].squeeze()

        return pred_i, target_i

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

    def __init__(self, prot_id: str, target: np.array, predictions: np.array):
        assert target.shape == predictions.shape, "The target and prediction arrays need to have the same shape"
        self.prot_id = prot_id
        self.target = target
        self.predictions = predictions

    def add_predictions(self, predictions):
        self.predictions = np.add(self.predictions, np.around(predictions, 3))

    def normalize_predictions(self, norm_factor):
        self.predictions = np.around(self.predictions / norm_factor, 3)

    def __len__(self):
        return self.target.shape[1]


class Results(object):
    def __init__(self):
        self.results_dict: Dict[str, ProteinResult] = {}

    def merge(self, results_dict: Results):
        self.results_dict = {**self.results_dict, **results_dict}

    def keys(self):
        return self.results_dict.keys()

    def __getitem__(self, item) -> ProteinResult:
        return self.results_dict[item]

    def __setitem__(self, key: str, value: ProteinResult):
        self.results_dict[key] = value

    def __len__(self):
        len(self.results_dict)

    def to_df(self) -> pd.DataFrame:
        prot_ids = []
        positions = []
        targets = []
        predictions = []
        ligands = []

        for prot_id, prot_result in self.results_dict.items():
            target = prot_result.target.flatten()
            targets.extend(target.tolist())
            prediction = prot_result.predictions.flatten()
            predictions.extend(prediction.tolist())
            positions.extend(list(range(len(prot_result))) * 3)
            prot_ids.extend([prot_id] * len(prot_result) * 3)
            ligands.extend(([0] * len(prot_result)) + ([1] * len(prot_result)) + ([2] * len(prot_result)))

        return pd.DataFrame(zip(prot_ids, positions, ligands, targets, predictions),
                            columns=['protd_id', 'position', 'ligand', 'target', 'prediction'])


class TrainEpoch(object):
    def __init__(self, _id: int):
        self._id = _id
        self.features = None
        self.pred = None
        self.target = None
        self.metrics = {
            "loss": 0,
            "loss_count": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "acc": 0,
            "prec": 0,
            "rec": 0,
            "f1": 0,
            "mcc": 0
        }
        self.sigm = torch.nn.Sigmoid()

    def add_batch(self, padding_batch: torch.tensor, pred_batch: torch.tensor, target_batch: torch.tensor):
        for idx, padding in enumerate(padding_batch):
            # remove padded positions to calculate tp, fp, tn, fn
            pred, target = General.remove_padded_positions(pred_batch[idx], target_batch[idx], padding)
            pred = self.sigm(pred)
            tp, fp, tn, fn = self.evaluate_per_residue_torch(pred, target)
            acc, prec, rec, f1, mcc = self.calc_performance_measurements(tp, fp, tn, fn)
            self.metrics["tp"] += tp
            self.metrics["fp"] += fp
            self.metrics["tn"] += tn
            self.metrics["fn"] += fn

            self.metrics["acc"] += acc
            self.metrics["prec"] += prec
            self.metrics["rec"] += rec
            self.metrics["f1"] += f1
            self.metrics["mcc"] += mcc

    def normalize(self):
        self.metrics["loss"] = self.metrics["loss"] / (self.metrics["loss_count"] * 3)

        self.metrics["acc"] /= self.metrics["loss_count"]
        self.metrics["prec"] /= self.metrics["loss_count"]
        self.metrics["rec"] /= self.metrics["loss_count"]
        self.metrics["f1"] /= self.metrics["loss_count"]
        self.metrics["mcc"] /= self.metrics["loss_count"]

    def __str__(self):
        return "Loss: {:.3f}, Prec: {:.3f}, Recall: {:.3f}, F1: {:.3f}, MCC: {:.3f}".format(self.metrics["loss"],
                                                                                            self.metrics["prec"],
                                                                                            self.metrics["rec"],
                                                                                            self.metrics["f1"],
                                                                                            self.metrics["mcc"]) + \
               "\n" + 'TP: {}, FP: {}, TN: {}, FN: {}'.format(
            self.metrics["tp"], self.metrics["fp"], self.metrics["tn"], self.metrics["fn"])

    @staticmethod
    def evaluate_per_residue_torch(prediction, target):
        """Calculate tp, fp, tn, fn for tensor"""
        # reduce prediction & target to one dimension
        prediction = prediction.t()
        target = target.t()
        prediction = torch.sum(torch.ge(prediction, 0.5), 1)
        target = torch.sum(torch.ge(target, 0.5), 1)

        # get confusion matrix
        tp = torch.sum(torch.ge(prediction, 0.5) * torch.ge(target, 0.5)).item()
        tn = torch.sum(torch.lt(prediction, 0.5) * torch.lt(target, 0.5)).item()
        fp = torch.sum(torch.ge(prediction, 0.5) * torch.lt(target, 0.5)).item()
        fn = torch.sum(torch.lt(prediction, 0.5) * torch.ge(target, 0.5)).item()

        return tp, fp, tn, fn

    @staticmethod
    def calc_performance_measurements(tp, fp, tn, fn):
        """Calculate precision, recall, f1, mcc, and accuracy"""

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


class TrainPerformance(object):
    def __init__(self):
        self.train_metrics: List[dict] = []
        self.validate_metrics: List[TrainEpoch] = []

    def add_epoch_performance(self, train_epoch: TrainEpoch, validate_epoch: TrainEpoch):
        self.train_metrics.append(train_epoch)
        self.validate_metrics.append(validate_epoch)

    def plot(self, dir: str):
        pass
        # TODO
        # df = self.to_df()
        #
        # # plt.plot([1, 2, 3, 4])
        # # plt.ylabel('some numbers')
        # # plt.show()

    def to_csv(self, file: str):
        self.to_df().to_csv(file)

    def to_df(self) -> pd.DataFrame:
        validation_df = self.__to_df(self.validate_metrics, "val")
        train_df = self.__to_df(self.train_metrics, "train")
        df = pd.concat([validation_df, train_df])
        df['epoch'] = list(range(len(df)))
        df.set_index("epoch", inplace=True)
        return df

    @staticmethod
    def __to_df(pe_list: List[TrainEpoch], name: str):
        metrics = {}
        for key in pe_list[0].metrics.keys():
            metrics[key] = []

        for pe in pe_list:
            for key, value in pe.metrics.items():
                metrics[key].append(value)

        keys = metrics.keys()
        values = [metrics[key] for key in keys]
        column_names = [key + "_" + name for key in keys]

        return pd.DataFrame(zip(*values),
                            columns=column_names)
