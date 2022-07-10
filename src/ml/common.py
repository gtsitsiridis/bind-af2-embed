from __future__ import annotations

import torch
import numpy as np
from typing import List, Union
import pandas as pd
from typing import Dict
from logging import getLogger
from torch import Tensor
from data.dataset import BindAnnotation
from pathlib import Path
import math
from scipy.stats import t

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

    def prediction_to_ri(self, cutoff: float) -> np.array:
        return np.abs(self._predictions - cutoff + 1e-10) / (cutoff + 1e-10) * 9

    def prediction_to_labels(self, cutoff: float) -> np.array:
        pred_copy = np.zeros(self._predictions.shape, dtype=np.int32)
        pred_copy[self._predictions >= cutoff] = 1
        return pred_copy

    def to_df(self, cutoff: Union[float, List[float]]) -> pd.DataFrame:
        if not isinstance(cutoff, list):
            cutoff = [cutoff]

        dfs = []
        for cutoff_i in cutoff:
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
            cutoffs.extend([cutoff_i] * len(self) * 4)
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
            ris.extend(self.prediction_to_ri(cutoff=cutoff_i).flatten())
            predictions.extend(self.prediction_to_labels(cutoff=cutoff_i).flatten())
            losses.extend(self._loss.flatten())
            dfs.append(pd.DataFrame(
                zip(tags, prot_ids, positions, ligands, targets, predictions, cutoffs, ris, probabilities, losses),
                columns=['tag', 'prot_id', 'position', 'ligand', 'target', 'prediction', 'cutoff', 'ri',
                         'prob', 'loss']))

        return pd.concat(dfs, axis=0)

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

    def to_df(self, cutoff: Union[float, List[float]]) -> pd.DataFrame:
        return pd.concat(list(map(lambda prot_result: prot_result.to_df(cutoff=cutoff),
                                  list(self.results_dict.values()))),
                         axis=0)

    def get_performance(self, cutoff: float, is_train: bool = False, tag_value: str = None) -> Performance:
        df = self.to_df(cutoff=cutoff)
        if tag_value is not None:
            df = df[df['tag'] == tag_value]
        else:
            df.tag = 'no-tag'
        return Performance.df_to_performance(df, is_train=is_train)

    def get_performance_per_cutoff(self, cutoffs: List[float]) -> Dict[float, Performance]:
        results = {}
        for cutoff in cutoffs:
            results[cutoff] = self.get_performance(cutoff=cutoff)
        return results


class Performance(object):
    def __init__(self, metrics: dict):
        self._metrics = metrics

    def keys(self):
        return self._metrics.keys()

    def to_dict(self):
        return self._metrics

    def __getitem__(self, item):
        return self._metrics[item]

    def to_str(self, tag: str) -> str:
        return "Loss: {:.3f}, Prec: {:.3f}, Recall: {:.3f}, F1: {:.3f}, MCC: {:.3f}".format(self["loss_" + tag],
                                                                                            self["prec_" + tag],
                                                                                            self["rec_" + tag],
                                                                                            self["f1_" + tag],
                                                                                            self["mcc_" + tag])

    @staticmethod
    def calc_performance_measurements(df: pd.DataFrame, tag: str, is_train: bool,
                                      ligand_check: bool = True) -> dict:
        """Calculate precision, recall, f1, mcc, and accuracy"""
        columns = {'prot_id', 'position', 'prediction', 'target'}
        assert columns.issubset(set(df.columns)), \
            'the dataframe should include the following columns: ' + str(columns)
        assert not ligand_check or len(
            df.ligand.unique()) == 1, 'You are trying to calculate performance measurements on multiple ligands!'

        def get_mean_ci(vec):
            """
            Calculate mean and 95% CI for a given vector
            :param vec: vector
            :return: mean and ci
            """
            mean = round(np.average(vec), 3)
            if len(vec) > 1:
                ci = round(np.std(vec) / math.sqrt(len(vec)) * t.ppf((1 + 0.95) / 2, len(vec)), 3)
            else:
                ci = 0

            return {'mean': mean, 'ci': ci}

        metrics = {}

        if is_train:
            # if in train mode, calculate stats based on all values (don't group per protein)
            metrics.update(Performance._calc_performance_measurements(df, tag=tag))
        else:
            # covonebind
            df1 = df.groupby('prot_id')[['prediction', 'target']].sum()
            metrics['covonebind_' + tag] = len(df1[df1.prediction > 0]) / (len(df1))

            # get protein based measurements
            metrics_df = df.groupby('prot_id'). \
                apply(lambda df_prot: pd.Series(Performance._calc_performance_measurements(df_prot, tag=tag)))

            # means with CI
            metrics_df = metrics_df.apply(lambda metric: pd.Series(get_mean_ci(metric)), axis=0)
            means = metrics_df.loc['mean']
            cis = metrics_df.loc['ci']
            cis.index = cis.index.map(lambda x: x + '_ci')
            metrics.update(means.to_dict())
            metrics.update(cis.to_dict())

        return metrics

    @staticmethod
    def _calc_performance_measurements(df: pd.DataFrame, tag: str) -> dict:
        columns = {'prot_id', 'target', 'prediction'}
        assert columns.issubset(set(df.columns)), \
            'the dataframe should include the following columns: ' + str(columns) + ". Given: " + str(df.columns)

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
        metrics = {'acc_' + tag: acc, 'prec_' + tag: prec, 'rec_' + tag: recall, 'f1_' + tag: f1, 'mcc_' + tag: mcc}
        return metrics

    @staticmethod
    def calc_cross_predictions(df: pd.DataFrame) -> dict:
        """Calculate cross-predictions"""
        columns = {'prot_id', 'target', 'prediction', 'ligand'}
        assert columns.issubset(set(df.columns)), \
            'the dataframe should include the following columns: ' + str(columns) + ". Given: " + str(df.columns)
        cross_predictions = {ligand + '_cross_pred': 0 for ligand in df.ligand.unique()}
        cross_predictions['true_pred'] = 0
        cross_predictions['total_cross_pred'] = 0

        def get_cross_predictions(df_position):
            # not that efficient but whatever
            for i in range(0, len(df_position)):
                row_i = df_position.iloc[i]
                ligand_i = row_i['ligand']
                prediction_i = row_i['prediction']
                target_i = row_i['target']
                if prediction_i == target_i == 1:
                    cross_predictions['true_pred'] += 1
                elif prediction_i != target_i:
                    # check for cross_predictions
                    for j in range(i + 1, len(df_position)):
                        row_j = df_position.iloc[j]
                        ligand_j = row_j['ligand']
                        prediction_j = row_j['prediction']
                        target_j = row_j['target']
                        if prediction_i == 1 and target_j == 1:
                            cross_predictions[ligand_i + '_cross_pred'] += 1
                            cross_predictions['total_cross_pred'] += 1
                        elif target_i == 1 and prediction_j == 1:
                            cross_predictions[ligand_j + '_cross_pred'] += 1
                            cross_predictions['total_cross_pred'] += 1

        df.groupby(['prot_id', 'position'])[['prediction', 'target', 'ligand']].apply(
            get_cross_predictions)

        return cross_predictions

    @staticmethod
    def df_to_performance(df: pd.DataFrame, is_train: bool):
        metrics = {}

        # sanity checks
        assert len(df.cutoff.unique()) == 1, 'You are trying to compute the performance of multiple cutoffs'
        assert len(df.tag.unique()) == 1, 'You are trying to compute the performance of multiple tags'

        ligands = set(df.ligand.unique())
        ligands_nobinding = ligands - {'binding'}

        # ligand metrics
        for ligand in ligands:
            metrics.update(
                Performance.calc_performance_measurements(df=df[df.ligand == ligand], tag=ligand, is_train=is_train))

        # total metrics (used for training)
        metrics['loss_total'] = df.loss.mean()
        # all predictions (including different ligands)
        metrics.update(Performance.calc_performance_measurements(df=df, tag='total', is_train=is_train,
                                                                 ligand_check=False))

        # these metrics will slow down the training
        if not is_train:
            # cross-predictions
            cross_predictions = Performance.calc_cross_predictions(df[df.ligand.isin(ligands_nobinding)])
            metrics.update(cross_predictions)

            # binding 2 (by merging the ligand outcomes)
            df_merged = df[df.ligand.isin(ligands_nobinding)].groupby(['prot_id', 'position'])[
                ['prediction', 'target']].max().reset_index()
            metrics.update(Performance.calc_performance_measurements(df=df_merged, tag='binding_2', is_train=is_train))

        return Performance(metrics=metrics)


class PerformanceMap(object):
    def __init__(self):
        self._metrics = {'tag': []}
        self._cross_predictions = {'tag': []}

    def get_performance(self, idx: int) -> Performance:
        assert 0 <= idx < len(self), 'idx is out of range'
        return Performance({k: v[idx] for k, v in self._metrics.items() if k != 'tag'})

    def append_performance(self, performance: Performance, tag: str):
        metrics = self._metrics
        metrics['tag'].append(tag)
        for k in performance.keys():
            if k not in self._metrics:
                metrics[k] = [performance[k]]
            else:
                metrics[k].append(performance[k])

    def keys(self):
        return self._metrics.keys()

    def to_df(self):
        return pd.DataFrame(self._metrics)

    def __getitem__(self, item):
        return self._metrics[item]

    def __len__(self):
        return len(self._metrics['loss'])
